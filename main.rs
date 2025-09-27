#![doc = r#"
# TestHashers: Comprehensive Hash Function Quality Testing

This toolkit implements rigorous hash function quality tests inspired by SMHasher and SMHasher3,
focusing on avalanche effect, differential analysis, and distribution quality.

Unlike simple collision counting, these tests reveal fundamental weaknesses in hash function design
by systematically testing how input bit patterns affect output distribution.

## Test Types

- **Avalanche Test**: Measures how single-bit input changes affect output distribution
- **Differential Test**: Detects patterns where minimal input differences cause predictable output patterns
- **Distribution Test**: Evaluates statistical uniformity of hash outputs
- **N-gram Analysis**: Tests quality across different input sizes using continuous buffer sampling

## Usage

```sh
# Test all hash functions
RUSTFLAGS="-C target-cpu=native" cargo run --release

# Stress test all hash functions extensively
RUSTFLAGS="-C target-cpu=native" cargo run --release -- --samples 10_000_000_000

# Test specific hash functions
RUSTFLAGS="-C target-cpu=native" cargo run --release -- --hash gxhash --hash xxhash3

# List available hash functions
cargo run -- --list-hashes

# Test with custom parameters
RUSTFLAGS="-C target-cpu=native" cargo run --release -- --samples 50000 --ngrams 8,16,32
```

## Quality Metrics

- **Avalanche Score**: Percentage of output bits that change on single input bit flip (ideal: ~50%)
- **Bias Score**: Maximum deviation from 50% for any output bit (good: <1%, excellent: <0.1%)
- **Differential Collisions**: Unexpected collisions in sparse key patterns
- **Chi-square**: Statistical measure of output distribution uniformity

"#]

mod hash_functions;

use clap::Parser;
use fork_union as fu;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha20Rng;

use hash_functions::{
    get_all_hash_functions, get_available_hash_names, get_hash_functions_by_names, HashFunction,
};

/// Trait for hash values that can be analyzed for quality
pub trait HashValue: Copy + PartialEq + std::fmt::Debug {
    fn xor(self, other: Self) -> Self;
    fn count_ones(self) -> u32;
    fn total_bits() -> u32;
    fn to_u64(self) -> u64;
    fn from_bytes(bytes: &[u8]) -> Self;
}

impl HashValue for u32 {
    fn xor(self, other: Self) -> Self {
        self ^ other
    }
    fn count_ones(self) -> u32 {
        self.count_ones()
    }
    fn total_bits() -> u32 {
        32
    }
    fn to_u64(self) -> u64 {
        self as u64
    }
    fn from_bytes(bytes: &[u8]) -> Self {
        if bytes.len() >= 4 {
            u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]])
        } else {
            let mut buf = [0u8; 4];
            buf[..bytes.len()].copy_from_slice(bytes);
            u32::from_le_bytes(buf)
        }
    }
}

impl HashValue for u64 {
    fn xor(self, other: Self) -> Self {
        self ^ other
    }
    fn count_ones(self) -> u32 {
        self.count_ones()
    }
    fn total_bits() -> u32 {
        64
    }
    fn to_u64(self) -> u64 {
        self
    }
    fn from_bytes(bytes: &[u8]) -> Self {
        if bytes.len() >= 8 {
            u64::from_le_bytes([
                bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
            ])
        } else {
            let mut buf = [0u8; 8];
            buf[..bytes.len()].copy_from_slice(bytes);
            u64::from_le_bytes(buf)
        }
    }
}

/// Results of avalanche analysis
#[derive(Debug)]
pub struct AvalancheResults {
    pub average_avalanche: f64,   // Average % of output bits that change
    pub worst_bias: f64,          // Maximum deviation from 50% for any bit
    pub best_bias: f64,           // Minimum deviation from 50% for any bit
    pub variance: f64,            // Variance in avalanche across bit positions
    pub per_bit_scores: Vec<f64>, // Avalanche score for each output bit
    pub total_tests: usize,       // Number of bit-flip tests performed
}

/// Results of differential key analysis
#[derive(Debug)]
pub struct DifferentialResults {
    pub expected_collisions: f64,
    pub actual_collisions: usize,
    pub collision_ratio: f64,
    pub worst_pattern: Option<Vec<usize>>, // Bit positions of worst differential
    pub total_tests: usize,
}

/// Results of distribution analysis
#[derive(Debug)]
pub struct DistributionResults {
    pub chi_square: f64,
    pub p_value: f64,
    pub uniformity_score: f64,
    pub bucket_count: usize,
}

/// Comprehensive quality test results
#[derive(Debug)]
pub struct QualityResults {
    pub function_name: String,
    pub avalanche: AvalancheResults,
    pub differential: DifferentialResults,
    pub distribution: DistributionResults,
}

/// Generate a large continuous random buffer for n-gram testing
pub fn generate_test_buffer(buffer_size: usize, seed: u64) -> Vec<u8> {
    let mut rng = ChaCha20Rng::seed_from_u64(seed);
    let mut buffer = vec![0u8; buffer_size];
    rng.fill(&mut buffer[..]);
    buffer
}

/// Iterator over n-grams in a buffer (zero-copy)
pub struct NGramIterator<'a> {
    buffer: &'a [u8],
    ngram_length: usize,
    position: usize,
}

impl<'a> NGramIterator<'a> {
    pub fn new(buffer: &'a [u8], ngram_length: usize) -> Self {
        Self {
            buffer,
            ngram_length,
            position: 0,
        }
    }
}

impl<'a> Iterator for NGramIterator<'a> {
    type Item = &'a [u8];

    fn next(&mut self) -> Option<Self::Item> {
        if self.position + self.ngram_length <= self.buffer.len() {
            let ngram = &self.buffer[self.position..self.position + self.ngram_length];
            self.position += 1;
            Some(ngram)
        } else {
            None
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self
            .buffer
            .len()
            .saturating_sub(self.position + self.ngram_length - 1);
        (remaining, Some(remaining))
    }
}

/// Optimized parallel avalanche analysis using fork_union's for_n
pub fn test_avalanche<H, F>(
    thread_pool: &mut fu::ThreadPool,
    hash_function: F,
    ngram_size: usize,
    num_samples: usize,
) -> AvalancheResults
where
    H: HashValue + Send + Sync,
    F: Fn(&[u8]) -> H + Send + Sync,
{
    // Generate test data buffer once
    let test_buffer_size = ngram_size + num_samples;
    let test_buffer = generate_test_buffer(test_buffer_size, 42);
    let num_output_bits = H::total_bits() as usize;
    let num_input_bits_per_ngram = ngram_size * 8;

    // Pre-allocate thread-local resources to avoid contention
    let num_worker_threads = thread_pool.threads();
    let mut worker_scratch_buffers: Vec<Vec<u8>> = (0..num_worker_threads)
        .map(|_| vec![0u8; ngram_size])
        .collect();

    // Thread-local bit flip counters (eliminates atomic contention)
    let mut worker_bit_flip_counters: Vec<Vec<usize>> = (0..num_worker_threads)
        .map(|_| vec![0; num_output_bits])
        .collect();

    // Safety: Each thread accesses only its own dedicated buffer and counter arrays
    let scratch_buffers_base_ptr = worker_scratch_buffers.as_mut_ptr() as usize;
    let bit_counters_base_ptr = worker_bit_flip_counters.as_mut_ptr() as usize;

    thread_pool.for_n(num_samples, |worker_context| {
        let ngram_start_index = worker_context.task_index;
        let worker_thread_id = worker_context.thread_index;

        // Early return for out-of-bounds samples
        if ngram_start_index + ngram_size > test_buffer.len() {
            return;
        }

        let current_ngram = &test_buffer[ngram_start_index..ngram_start_index + ngram_size];
        let baseline_hash = hash_function(current_ngram);

        // Get thread-local resources (zero contention)
        let scratch_buffer =
            unsafe { &mut *(scratch_buffers_base_ptr as *mut Vec<u8>).add(worker_thread_id) };
        let bit_flip_counters =
            unsafe { &mut *(bit_counters_base_ptr as *mut Vec<usize>).add(worker_thread_id) };

        // Test each input bit flip for avalanche effect
        for input_bit_position in 0..num_input_bits_per_ngram {
            let target_byte_index = input_bit_position / 8;
            let target_bit_index = input_bit_position % 8;

            // Create modified n-gram with single bit flipped
            scratch_buffer.copy_from_slice(current_ngram);
            scratch_buffer[target_byte_index] ^= 1 << target_bit_index;

            let perturbed_hash = hash_function(scratch_buffer);
            let hash_difference = baseline_hash.xor(perturbed_hash);
            let difference_bits = hash_difference.to_u64();

            // Count affected output bits (thread-local, no contention)
            for output_bit_position in 0..num_output_bits {
                if (difference_bits >> output_bit_position) & 1 == 1 {
                    bit_flip_counters[output_bit_position] += 1;
                }
            }
        }
    });

    // Aggregate results from all worker threads
    let mut total_bit_flip_counts = vec![0usize; num_output_bits];

    for worker_counters in &worker_bit_flip_counters {
        for (output_bit_index, &flip_count) in worker_counters.iter().enumerate() {
            total_bit_flip_counts[output_bit_index] += flip_count;
        }
    }

    // Calculate total avalanche tests performed
    let valid_ngram_count = num_samples.min(test_buffer.len().saturating_sub(ngram_size - 1));
    let total_avalanche_tests = valid_ngram_count * num_input_bits_per_ngram;

    // Calculate avalanche percentages for each output bit
    let avalanche_percentages: Vec<f64> = total_bit_flip_counts
        .iter()
        .map(|&flip_count| (flip_count as f64 / total_avalanche_tests as f64) * 100.0)
        .collect();

    let mean_avalanche_percentage =
        avalanche_percentages.iter().sum::<f64>() / num_output_bits as f64;

    // Calculate bias from ideal 50% avalanche
    let bias_values: Vec<f64> = avalanche_percentages
        .iter()
        .map(|&percentage| (percentage - 50.0).abs())
        .collect();

    let maximum_bias = bias_values.iter().fold(0.0f64, |acc, &bias| acc.max(bias));
    let minimum_bias = bias_values
        .iter()
        .fold(f64::INFINITY, |acc, &bias| acc.min(bias));

    // Calculate variance in avalanche across output bits
    let avalanche_variance = avalanche_percentages
        .iter()
        .map(|percentage| (percentage - mean_avalanche_percentage).powi(2))
        .sum::<f64>()
        / num_output_bits as f64;

    AvalancheResults {
        average_avalanche: mean_avalanche_percentage,
        worst_bias: maximum_bias,
        best_bias: minimum_bias,
        variance: avalanche_variance,
        per_bit_scores: avalanche_percentages,
        total_tests: total_avalanche_tests,
    }
}

/// Optimized parallel differential pattern analysis
pub fn test_differential<H, F>(
    thread_pool: &mut fu::ThreadPool,
    hash_function: F,
    ngram_size: usize,
    num_bits_to_flip: usize,
    num_samples: usize,
) -> DifferentialResults
where
    H: HashValue + Send + Sync,
    F: Fn(&[u8]) -> H + Send + Sync,
{
    use std::sync::atomic::{AtomicUsize, Ordering};

    // Generate test data buffer once
    let test_buffer_size = ngram_size + num_samples;
    let test_buffer = generate_test_buffer(test_buffer_size, 123);

    // Pre-allocate thread-local scratch buffers
    let num_worker_threads = thread_pool.threads();
    let mut worker_scratch_buffers: Vec<Vec<u8>> = (0..num_worker_threads)
        .map(|_| vec![0u8; ngram_size])
        .collect();

    // Atomic collision counter (thread-safe)
    let collision_counter = AtomicUsize::new(0);

    // Safety: Each thread accesses only its own dedicated buffer
    let scratch_buffers_base_ptr = worker_scratch_buffers.as_mut_ptr() as usize;

    thread_pool.for_n(num_samples, |worker_context| {
        let ngram_start_index = worker_context.task_index;
        let worker_thread_id = worker_context.thread_index;

        // Early return for out-of-bounds samples
        if ngram_start_index + ngram_size > test_buffer.len() {
            return;
        }

        let current_ngram = &test_buffer[ngram_start_index..ngram_start_index + ngram_size];
        let baseline_hash = hash_function(current_ngram);

        // Get thread-local scratch buffer
        let scratch_buffer =
            unsafe { &mut *(scratch_buffers_base_ptr as *mut Vec<u8>).add(worker_thread_id) };

        // Create variant with specified number of bits flipped
        scratch_buffer.copy_from_slice(current_ngram);

        // Flip the first num_bits_to_flip bits
        for bit_index in 0..num_bits_to_flip {
            let target_byte_index = bit_index / 8;
            let target_bit_index = bit_index % 8;
            if target_byte_index < ngram_size {
                scratch_buffer[target_byte_index] ^= 1 << target_bit_index;
            }
        }

        let perturbed_hash = hash_function(scratch_buffer);

        // Check for collision (thread-safe)
        if baseline_hash == perturbed_hash {
            collision_counter.fetch_add(1, Ordering::Relaxed);
        }
    });

    let total_collisions = collision_counter.load(Ordering::Relaxed);

    // Calculate expected collision rate based on hash space size
    let hash_space_size = 2.0_f64.powi(H::total_bits() as i32);
    let expected_collision_rate = (num_samples as f64) / hash_space_size;
    let collision_ratio = total_collisions as f64 / expected_collision_rate.max(1e-10);

    DifferentialResults {
        expected_collisions: expected_collision_rate,
        actual_collisions: total_collisions,
        collision_ratio,
        worst_pattern: None, // Could be enhanced to track problematic bit patterns
        total_tests: num_samples,
    }
}

/// Optimized parallel distribution uniformity analysis
pub fn test_distribution<H, F>(
    thread_pool: &mut fu::ThreadPool,
    hash_function: F,
    ngram_size: usize,
    num_samples: usize,
    num_buckets: usize,
) -> DistributionResults
where
    H: HashValue + Send + Sync,
    F: Fn(&[u8]) -> H + Send + Sync,
{
    // Generate test data buffer once
    let test_buffer_size = ngram_size + num_samples;
    let test_buffer = generate_test_buffer(test_buffer_size, 789);

    // Pre-allocate thread-local bucket counters to avoid contention
    let num_worker_threads = thread_pool.threads();
    let mut worker_bucket_counters: Vec<Vec<usize>> = (0..num_worker_threads)
        .map(|_| vec![0; num_buckets])
        .collect();

    // Safety: Each thread accesses only its own dedicated counter array
    let bucket_counters_base_ptr = worker_bucket_counters.as_mut_ptr() as usize;

    thread_pool.for_n(num_samples, |worker_context| {
        let ngram_start_index = worker_context.task_index;
        let worker_thread_id = worker_context.thread_index;

        // Early return for out-of-bounds samples
        if ngram_start_index + ngram_size > test_buffer.len() {
            return;
        }

        let current_ngram = &test_buffer[ngram_start_index..ngram_start_index + ngram_size];
        let hash_value = hash_function(current_ngram);

        // Get thread-local bucket counters (zero contention)
        let bucket_counters =
            unsafe { &mut *(bucket_counters_base_ptr as *mut Vec<usize>).add(worker_thread_id) };

        // Distribute hash into appropriate bucket
        let target_bucket = (hash_value.to_u64() % num_buckets as u64) as usize;
        bucket_counters[target_bucket] += 1;
    });

    // Aggregate results from all worker threads
    let mut total_bucket_counts = vec![0usize; num_buckets];

    for worker_counters in &worker_bucket_counters {
        for (bucket_index, &count) in worker_counters.iter().enumerate() {
            total_bucket_counts[bucket_index] += count;
        }
    }

    // Calculate chi-square statistic for uniformity test
    let expected_count_per_bucket = num_samples as f64 / num_buckets as f64;
    let chi_square_statistic: f64 = total_bucket_counts
        .iter()
        .map(|&observed_count| {
            let deviation = observed_count as f64 - expected_count_per_bucket;
            (deviation * deviation) / expected_count_per_bucket
        })
        .sum();

    // Simplified p-value calculation (chi-square approximation)
    let degrees_of_freedom = num_buckets - 1;
    let p_value_estimate = 1.0 - (chi_square_statistic / degrees_of_freedom as f64).min(1.0);

    // Calculate uniformity score (0-100, higher = more uniform)
    let maximum_relative_deviation = total_bucket_counts
        .iter()
        .map(|&count| {
            ((count as f64 - expected_count_per_bucket) / expected_count_per_bucket).abs()
        })
        .fold(0.0f64, |acc, deviation| acc.max(deviation));

    let uniformity_percentage = ((1.0 - maximum_relative_deviation.min(1.0)) * 100.0).max(0.0);

    DistributionResults {
        chi_square: chi_square_statistic,
        p_value: p_value_estimate,
        uniformity_score: uniformity_percentage,
        bucket_count: num_buckets,
    }
}

/// Test hash functions across multiple n-gram sizes and print tabular results
fn test_hash_functions_tabular(
    hash_functions: &[Box<dyn HashFunction>],
    ngram_sizes: &[usize],
    samples_per_size: usize,
    verbose: bool,
) {
    // Collect results for all hash functions and n-gram sizes
    struct TestResult {
        hash_name: String,
        ngram_size: usize,
        avalanche_bias: f64,
        differential_collisions: usize,
        distribution_score: f64,
        ngrams_tested: usize,
    }

    let mut all_results = Vec::new();

    // Create thread pool once for all tests
    let num_cores = fu::count_logical_cores();
    let mut pool = fu::spawn(num_cores);

    for ngram_size in ngram_sizes {
        // Cap samples based on combinatorial space for small n-grams
        let max_possible_ngrams = if *ngram_size <= 5 {
            1usize << ngram_size * 8
        } else {
            samples_per_size
        };

        let effective_samples = samples_per_size.min(max_possible_ngrams);
        if effective_samples < samples_per_size {
            println!(
                "\nTesting {}-grams with {} samples (capped from {} - only 2^{} possible combinations) using {} cores",
                ngram_size, effective_samples, samples_per_size, ngram_size * 8, num_cores
            );
        } else {
            println!(
                "\nTesting {}-grams with {} samples each... (using {} cores)",
                ngram_size, effective_samples, num_cores
            );
        }

        for hash_func in hash_functions {
            print!("  Testing {}...", hash_func.name());

            // Run tests based on bit width
            let (avalanche, differential, distribution, actual_samples) = if hash_func.bits() == 64
            {
                let aval = test_avalanche(
                    &mut pool,
                    |d| hash_func.hash(d),
                    *ngram_size,
                    effective_samples,
                );
                let diff = test_differential(
                    &mut pool,
                    |d| hash_func.hash(d),
                    *ngram_size,
                    1,
                    effective_samples,
                );
                let dist = test_distribution(
                    &mut pool,
                    |d| hash_func.hash(d),
                    *ngram_size,
                    effective_samples * 10,
                    1024,
                );
                let samples = effective_samples.min(dist.bucket_count * 10); // Actual samples tested
                (aval, diff, dist, samples)
            } else {
                let aval = test_avalanche(
                    &mut pool,
                    |d| hash_func.hash(d) as u32,
                    *ngram_size,
                    effective_samples,
                );
                let diff = test_differential(
                    &mut pool,
                    |d| hash_func.hash(d) as u32,
                    *ngram_size,
                    1,
                    effective_samples,
                );
                let dist = test_distribution(
                    &mut pool,
                    |d| hash_func.hash(d) as u32,
                    *ngram_size,
                    effective_samples * 10,
                    1024,
                );
                let samples = effective_samples.min(dist.bucket_count * 10);
                (aval, diff, dist, samples)
            };

            // Show detailed stats immediately if verbose mode
            if verbose {
                // Find best and worst performing bits
                let best_bit = avalanche
                    .per_bit_scores
                    .iter()
                    .enumerate()
                    .min_by(|a, b| (a.1 - 50.0).abs().partial_cmp(&(b.1 - 50.0).abs()).unwrap())
                    .unwrap();
                let worst_bit = avalanche
                    .per_bit_scores
                    .iter()
                    .enumerate()
                    .max_by(|a, b| (a.1 - 50.0).abs().partial_cmp(&(b.1 - 50.0).abs()).unwrap())
                    .unwrap();

                println!(" done");
                println!(
                    "    └─ Avalanche: avg={:.2}%, worst_bias={:.3}%, variance={:.2}",
                    avalanche.average_avalanche, avalanche.worst_bias, avalanche.variance
                );
                println!(
                    "    └─ Best bit {}: {:.2}%, Worst bit {}: {:.2}%",
                    best_bit.0, best_bit.1, worst_bit.0, worst_bit.1
                );
                println!(
                    "    └─ Distribution: score={:.1}%, chi²={:.1}",
                    distribution.uniformity_score, distribution.chi_square
                );
                println!(
                    "    └─ Differential: {} collisions (expected {:.2})",
                    differential.actual_collisions, differential.expected_collisions
                );
            } else {
                println!(" done");
            }

            all_results.push(TestResult {
                hash_name: hash_func.name().to_string(),
                ngram_size: *ngram_size,
                avalanche_bias: avalanche.worst_bias,
                differential_collisions: differential.actual_collisions,
                distribution_score: distribution.uniformity_score,
                ngrams_tested: actual_samples,
            });
        }
    }

    // Print tabular results
    println!("\n{}", "=".repeat(120));
    println!("COMPREHENSIVE HASH QUALITY ANALYSIS - TABULAR RESULTS");
    println!("{}", "=".repeat(120));

    // Print header
    println!(
        "{:<12} | {:>8} | {:>12} | {:>10} | {:>10} | {:>12} | {:>10}",
        "Hash", "N-gram", "N-grams", "Avalanche", "Quality", "Diff.Colls", "Dist.Score"
    );
    println!(
        "{:<12} | {:>8} | {:>12} | {:>10} | {:>10} | {:>12} | {:>10}",
        "Function", "Size", "Tested", "Bias %", "", "(expect 0)", "%"
    );
    println!("{}", "-".repeat(120));

    // Print results
    for result in &all_results {
        let avalanche_quality = if result.avalanche_bias < 0.1 {
            "EXCELLENT"
        } else if result.avalanche_bias < 0.5 {
            "VERY GOOD"
        } else if result.avalanche_bias < 1.0 {
            "GOOD"
        } else if result.avalanche_bias < 2.0 {
            "FAIR"
        } else {
            "POOR"
        };

        println!(
            "{:<12} | {:>8} | {:>12} | {:>10.3} | {:>10} | {:>12} | {:>10.1}",
            result.hash_name,
            result.ngram_size,
            result.ngrams_tested,
            result.avalanche_bias,
            avalanche_quality,
            result.differential_collisions,
            result.distribution_score
        );
    }

    println!("{}", "=".repeat(120));

    // Summary statistics by n-gram size
    println!("\nSUMMARY BY N-GRAM SIZE:");
    println!(
        "{:<10} | {:>15} | {:>15} | {:>15}",
        "N-gram", "Best Avalanche", "Worst Avalanche", "Avg Dist.Score"
    );
    println!("{}", "-".repeat(70));

    for ngram_size in ngram_sizes {
        let size_results: Vec<_> = all_results
            .iter()
            .filter(|r| r.ngram_size == *ngram_size)
            .collect();

        if !size_results.is_empty() {
            let best = size_results
                .iter()
                .min_by(|a, b| a.avalanche_bias.partial_cmp(&b.avalanche_bias).unwrap())
                .unwrap();
            let worst = size_results
                .iter()
                .max_by(|a, b| a.avalanche_bias.partial_cmp(&b.avalanche_bias).unwrap())
                .unwrap();
            let avg_dist = size_results
                .iter()
                .map(|r| r.distribution_score)
                .sum::<f64>()
                / size_results.len() as f64;

            println!(
                "{:<10} | {:>15} | {:>15} | {:>15.1}",
                ngram_size,
                format!("{} ({:.3}%)", best.hash_name, best.avalanche_bias),
                format!("{} ({:.3}%)", worst.hash_name, worst.avalanche_bias),
                avg_dist
            );
        }
    }
}

/// Command line arguments
#[derive(Parser)]
#[command(name = "testhashers")]
#[command(about = "Hash Function Quality Testing Laboratory")]
#[command(version)]
struct Args {
    /// Hash functions to test (can be specified multiple times)
    #[arg(long = "hash", value_name = "HASH")]
    hash_functions: Vec<String>,

    /// List all available hash functions
    #[arg(long = "list-hashes")]
    list_hashes: bool,

    /// Number of samples per n-gram size
    #[arg(long = "samples", default_value = "10000")]
    samples: usize,

    /// N-gram sizes to test (comma-separated)
    #[arg(long = "ngrams", default_value = "3,4,8,16,32,64,128")]
    ngram_sizes_str: String,

    /// Show detailed statistics for each individual test
    #[arg(long = "verbose", short = 'v')]
    verbose: bool,
}

fn parse_ngram_sizes(s: &str) -> Result<Vec<usize>, String> {
    s.split(',')
        .map(|size| {
            size.trim()
                .parse()
                .map_err(|e| format!("Invalid n-gram size '{}': {}", size, e))
        })
        .collect()
}

fn main() {
    let args = Args::parse();

    if args.list_hashes {
        println!("Available hash functions:");
        for name in get_available_hash_names() {
            println!("  {}", name);
        }
        return;
    }

    println!("TestHashers: Hash Function Quality Laboratory");
    println!("=============================================");
    println!("Testing hash functions on n-grams from continuous random buffer");
    println!();

    // Parse n-gram sizes
    let ngram_sizes = parse_ngram_sizes(&args.ngram_sizes_str).unwrap_or_else(|e| {
        eprintln!("Error parsing n-gram sizes: {}", e);
        std::process::exit(1);
    });

    // Select hash functions to test
    let hash_functions = if args.hash_functions.is_empty() {
        get_all_hash_functions()
    } else {
        let selected = get_hash_functions_by_names(&args.hash_functions);
        if selected.is_empty() {
            eprintln!("Error: No valid hash functions found. Available functions:");
            for name in get_available_hash_names() {
                eprintln!("  {}", name);
            }
            std::process::exit(1);
        }
        selected
    };

    println!("Configuration:");
    println!(
        "  Hash functions: {}",
        hash_functions
            .iter()
            .map(|h| h.name())
            .collect::<Vec<_>>()
            .join(", ")
    );
    println!("  N-gram sizes: {:?} bytes", ngram_sizes);
    println!("  Samples per size: {} n-grams", args.samples);
    println!(
        "  Total avalanche tests: {} bit flips per hash function",
        ngram_sizes
            .iter()
            .map(|&s| s * 8 * args.samples)
            .sum::<usize>()
    );

    test_hash_functions_tabular(&hash_functions, &ngram_sizes, args.samples, args.verbose);

    println!("\nNotes:");
    println!("- Avalanche bias: <0.1% (excellent), <0.5% (very good), <1.0% (good)");
    println!("- Distribution score: Higher is better (100% = perfect uniformity)");
    println!("- Differential collisions: Should be 0 or very close to 0");
    println!("- CRC32 is expected to have poor avalanche (not designed for it)");
}
