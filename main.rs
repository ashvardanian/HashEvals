#![doc = r#"
# HashEvals: Minimalistic Hash Function Quality Testing Suite

This toolkit implements rigorous hash function quality tests inspired by SMHasher and SMHasher3,
focusing on avalanche effect, differential analysis, and distribution quality.

Unlike simple collision counting, these tests reveal fundamental weaknesses in hash function design
by systematically testing how input bit patterns affect output distribution.

## Test Types

- **Avalanche Test**: Measures how single-bit input changes affect output distribution
- **Integral Collision Test**: Detects hash collisions using random N-byte inputs
- **Distribution Test**: Evaluates statistical uniformity of hash outputs
- **N-gram Analysis**: Tests quality across different input sizes using continuous buffer sampling

## Usage

```sh
# Make sure everything compiles and logs are shown
RUSTFLAGS="-C target-cpu=native" cargo run --release --samples 100 --verbose

# Stress-test all hash functions, outputting also to file... will take a while
time RUSTFLAGS="-C target-cpu=native" cargo run --release 2>&1 | tee results.txt

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
- **Integral Collisions**: Hash collisions on random N-byte little-endian integers
- **Chi-square**: Statistical measure of output distribution uniformity

"#]

mod hash_functions;

use std::fs::File;
use std::sync::atomic::{AtomicU64, Ordering};

use clap::Parser;
use fork_union as fu;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha20Rng;
use serde::{Deserialize, Serialize};
use tabled::{
    settings::{Alignment, Style},
    Table, Tabled,
};

use hash_functions::{
    get_all_hash_functions, get_available_hash_names, get_hash_functions_by_names, HashFunction,
};

fn format_thousands(n: usize) -> String {
    let s = n.to_string();
    let mut result = String::new();
    for (i, c) in s.chars().rev().enumerate() {
        if i > 0 && i % 3 == 0 {
            result.push('\'');
        }
        result.push(c);
    }
    result.chars().rev().collect()
}

fn format_duration(duration_ms: u64) -> String {
    if duration_ms < 1000 {
        format!("{}ms", format_thousands(duration_ms as usize))
    } else if duration_ms < 60_000 {
        format!("{:.1}s", duration_ms as f64 / 1000.0)
    } else {
        let total_seconds = duration_ms / 1000;
        let minutes = total_seconds / 60;
        let seconds = total_seconds % 60;
        format!("{}m {}s", format_thousands(minutes as usize), seconds)
    }
}

fn find_bias_leaders(results: &[&DetailedTestResult]) -> String {
    if results.is_empty() {
        return "N/A".to_string();
    }

    // Find the best (lowest) bias
    let best_bias = results
        .iter()
        .map(|r| r.avalanche_bias)
        .min_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .unwrap();

    // Find all results that have the best bias (ties)
    let leaders: Vec<String> = results
        .iter()
        .filter(|r| (r.avalanche_bias - best_bias).abs() < 1e-10) // Handle floating point precision
        .map(|r| r.hash_name.clone())
        .collect();

    leaders.join(", ")
}

fn find_speed_leaders(results: &[&DetailedTestResult]) -> String {
    if results.is_empty() {
        return "N/A".to_string();
    }

    // Find the best (lowest) duration
    let best_duration = results.iter().map(|r| r.duration_ms).min().unwrap();

    // Find all results that have the best duration (ties)
    let leaders: Vec<String> = results
        .iter()
        .filter(|r| r.duration_ms == best_duration)
        .map(|r| r.hash_name.clone())
        .collect();

    leaders.join(", ")
}

fn find_collision_leaders(results: &[&DetailedTestResult]) -> String {
    if results.is_empty() {
        return "N/A".to_string();
    }

    // Find the best (lowest) collision count
    let best_collisions = results
        .iter()
        .map(|r| r.total_collisions_count)
        .min()
        .unwrap();

    // Find all results that have the best collision count (ties)
    let leaders: Vec<String> = results
        .iter()
        .filter(|r| r.total_collisions_count == best_collisions)
        .map(|r| r.hash_name.clone())
        .collect();

    leaders.join(", ")
}

/// Trait for hash values that can be analyzed for quality
pub trait HashValue: Copy + PartialEq + std::fmt::Debug {
    fn xor(self, other: Self) -> Self;
    fn count_ones(self) -> u32;
    fn total_bits() -> u32;
    fn to_u64(self) -> u64;
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

/// Results of collision analysis
#[derive(Debug)]
pub struct CollisionResults {
    pub total_collisions: usize,
    pub first_collision_at: Option<usize>,
    pub expected_collision_at: usize, // Birthday paradox expectation
    pub total_tests: usize,
    pub collision_rate: f64, // Actual collision rate
}

/// Results of distribution analysis
#[derive(Debug)]
pub struct DistributionResults {
    pub chi_square: f64,
    pub bucket_count: usize,
}

/// Generate a large continuous random buffer for n-gram testing
pub fn generate_test_buffer(buffer_size: usize, seed: u64) -> Vec<u8> {
    let mut rng = ChaCha20Rng::seed_from_u64(seed);
    let mut buffer = vec![0u8; buffer_size];
    rng.fill(&mut buffer[..]);
    buffer
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

/// Test for integral collisions using sample-sized bitset and random inputs
pub fn test_integral_collisions<F>(
    thread_pool: &mut fu::ThreadPool,
    hash_function: F,
    ngram_size: usize,
    num_samples: usize,
) -> Option<CollisionResults>
where
    F: Fn(&[u8]) -> u64 + Send + Sync,
{
    // Only run for N-gram sizes ≤ 8 bytes
    if ngram_size > 8 {
        return None;
    }

    // Expected collision point using birthday paradox for sample space
    let expected_collision_at = ((num_samples as f64).sqrt() * 1.2533) as usize; // √(π/2) * √n

    // Generate random u64 values with only lower N bytes populated
    let mut rng = ChaCha20Rng::seed_from_u64(123456);
    let test_inputs: Vec<u64> = (0..num_samples)
        .map(|_| {
            let val = rng.random::<u64>();
            // Mask to only use lower N bytes
            let mask = if ngram_size >= 8 {
                u64::MAX
            } else {
                (1u64 << (ngram_size * 8)) - 1
            };
            val & mask
        })
        .collect();

    // Create atomic bitset of sample size
    let bitset_chunks = (num_samples + 63) / 64; // Round up to nearest u64
    let bitset: Vec<AtomicU64> = (0..bitset_chunks).map(|_| AtomicU64::new(0)).collect();

    // Atomic counters for collision detection
    let total_collisions = AtomicU64::new(0);
    let first_collision = AtomicU64::new(u64::MAX);

    thread_pool.for_n(num_samples, |worker_context| {
        let i = worker_context.task_index;
        let input_val = test_inputs[i];

        // Convert to bytes for hashing (little-endian, only N bytes)
        let input_bytes = input_val.to_le_bytes();
        let hash_val = hash_function(&input_bytes);

        // Map hash to bitset position using modulo
        let bit_index = (hash_val % num_samples as u64) as usize;
        let chunk_index = bit_index / 64;
        let bit_offset = bit_index % 64;
        let bit_mask = 1u64 << bit_offset;

        assert!(chunk_index < bitset.len());
        let old_value = bitset[chunk_index].fetch_or(bit_mask, Ordering::Relaxed);

        // Check if bit was already set (collision)
        if (old_value & bit_mask) == 0 {
            return;
        }
        total_collisions.fetch_add(1, Ordering::Relaxed);
        first_collision.fetch_min(i as u64, Ordering::Relaxed);
    });

    let total_collision_count = total_collisions.load(Ordering::Relaxed) as usize;
    let first_collision_idx = first_collision.load(Ordering::Relaxed);

    let first_collision_at = if first_collision_idx == u64::MAX {
        None
    } else {
        Some(first_collision_idx as usize)
    };

    let collision_rate = total_collision_count as f64 / num_samples as f64;

    Some(CollisionResults {
        total_collisions: total_collision_count,
        first_collision_at,
        expected_collision_at,
        total_tests: num_samples,
        collision_rate,
    })
}

/// Optimized parallel bucket distribution uniformity analysis
pub fn test_buckets_distribution<H, F>(
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

    DistributionResults {
        chi_square: chi_square_statistic,
        bucket_count: num_buckets,
    }
}

/// Detailed test result for individual N-gram sizes (used in verbose mode)
#[derive(Debug, Serialize, Deserialize, Tabled)]
struct DetailedTestResult {
    #[tabled(rename = "Hash Function")]
    hash_name: String,
    #[tabled(rename = "N-gram Size")]
    ngram_size: usize,
    #[tabled(rename = "N-grams Tested")]
    ngrams_tested: usize,
    #[tabled(rename = "Avalanche Bias")]
    avalanche_bias_display: String,
    #[tabled(rename = "Integral ⨳")]
    total_collisions: usize,
    #[tabled(rename = "1st Collision")]
    first_collision_display: String,
    #[tabled(rename = "Chi²")]
    distribution_score_display: String,
    #[serde(skip)]
    #[tabled(skip)]
    avalanche_bias: f64,
    #[serde(skip)]
    #[tabled(skip)]
    distribution_score: f64,
    #[serde(skip)]
    #[tabled(skip)]
    total_collisions_count: usize,
    #[serde(rename = "duration_ms")]
    #[tabled(skip)]
    duration_ms: u64,
}

/// Aggregated test result for main summary table and CSV export
#[derive(Debug, Serialize, Deserialize, Tabled)]
struct TestResult {
    #[tabled(rename = "Function")]
    hash_name: String,
    #[tabled(rename = "Avg.Bias")]
    avg_bias_display: String,
    #[tabled(rename = "Worst.Bias")]
    worst_bias_display: String,
    #[tabled(rename = "Integral ⨳")]
    total_collisions_display: String,
    #[tabled(rename = "Chi²")]
    avg_chi_square_display: String,
    #[tabled(rename = "Duration")]
    duration_display: String,
    #[serde(rename = "total_collisions")]
    #[tabled(skip)]
    total_collisions: usize,
    #[serde(rename = "duration_ms")]
    #[tabled(skip)]
    duration_ms: u64,
    #[serde(skip)]
    #[tabled(skip)]
    avg_bias: f64,
}

/// Aggregate detailed results by hash function
fn aggregate_results(detailed_results: &[DetailedTestResult]) -> Vec<TestResult> {
    use std::collections::HashMap;

    let mut hash_function_data: HashMap<String, Vec<&DetailedTestResult>> = HashMap::new();

    // Group results by hash function
    for result in detailed_results {
        hash_function_data
            .entry(result.hash_name.clone())
            .or_insert_with(Vec::new)
            .push(result);
    }

    // Aggregate each hash function's results
    let mut aggregated = Vec::new();

    for (hash_name, results) in hash_function_data {
        let avg_bias = results.iter().map(|r| r.avalanche_bias).sum::<f64>() / results.len() as f64;
        let worst_bias = results
            .iter()
            .map(|r| r.avalanche_bias)
            .fold(0.0f64, |acc, bias| acc.max(bias));
        let avg_chi_square =
            results.iter().map(|r| r.distribution_score).sum::<f64>() / results.len() as f64;

        // Sum total collisions across all n-gram sizes
        let total_collisions_sum = results
            .iter()
            .map(|r| r.total_collisions_count)
            .sum::<usize>();

        // Sum total duration across all n-gram sizes
        let total_duration_ms = results.iter().map(|r| r.duration_ms).sum::<u64>();

        aggregated.push(TestResult {
            hash_name,
            avg_bias_display: format!("{:.5}%", avg_bias),
            worst_bias_display: format!("{:.5}%", worst_bias),
            total_collisions_display: format_thousands(total_collisions_sum),
            avg_chi_square_display: format!("{:.3}", avg_chi_square),
            duration_display: format_duration(total_duration_ms),
            total_collisions: total_collisions_sum,
            duration_ms: total_duration_ms,
            avg_bias,
        });
    }

    // Sort by average bias (best first - lowest average bias is more statistically significant)
    aggregated.sort_by(|a, b| a.avg_bias.partial_cmp(&b.avg_bias).unwrap());

    aggregated
}

/// Export test results to CSV file
fn export_to_csv(results: &[TestResult], path: &str) -> Result<(), Box<dyn std::error::Error>> {
    let file = File::create(path)?;
    let mut writer = csv::Writer::from_writer(file);

    for result in results {
        writer.serialize(result)?;
    }

    writer.flush()?;
    Ok(())
}

/// Test hash functions across multiple n-gram sizes and print tabular results
fn test_hash_functions_tabular(
    hash_functions: &[Box<dyn HashFunction>],
    ngram_sizes: &[usize],
    samples_per_size: usize,
    verbose: bool,
    csv_output: Option<&str>,
) {
    // Collect detailed results for all hash functions and n-gram sizes
    let mut detailed_results = Vec::new();

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
                ngram_size, format_thousands(effective_samples), format_thousands(samples_per_size), ngram_size * 8, num_cores
            );
        } else {
            println!(
                "\nTesting {}-grams with {} samples each... (using {} cores)",
                ngram_size,
                format_thousands(effective_samples),
                num_cores
            );
        }

        for hash_func in hash_functions {
            print!("  Testing {}...", hash_func.name());

            // Start timing
            let start_time = std::time::Instant::now();

            // Run tests based on bit width
            let (avalanche, collisions_opt, distribution, actual_samples) =
                if hash_func.bits() == 64 {
                    let aval = test_avalanche(
                        &mut pool,
                        |d| hash_func.hash(d),
                        *ngram_size,
                        effective_samples,
                    );
                    let collisions_opt = test_integral_collisions(
                        &mut pool,
                        |d| hash_func.hash(d),
                        *ngram_size,
                        effective_samples,
                    );
                    let dist = test_buckets_distribution(
                        &mut pool,
                        |d| hash_func.hash(d),
                        *ngram_size,
                        effective_samples * 10,
                        1024,
                    );
                    let samples = effective_samples.min(dist.bucket_count * 10); // Actual samples tested
                    (aval, collisions_opt, dist, samples)
                } else {
                    let aval = test_avalanche(
                        &mut pool,
                        |d| hash_func.hash(d) as u32,
                        *ngram_size,
                        effective_samples,
                    );
                    let collisions_opt = test_integral_collisions(
                        &mut pool,
                        |d| hash_func.hash(d),
                        *ngram_size,
                        effective_samples,
                    );
                    let dist = test_buckets_distribution(
                        &mut pool,
                        |d| hash_func.hash(d) as u32,
                        *ngram_size,
                        effective_samples * 10,
                        1024,
                    );
                    let samples = effective_samples.min(dist.bucket_count * 10);
                    (aval, collisions_opt, dist, samples)
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
                    "    └─ Avalanche: avg={:.3}%, worst_bias={:.5}%, variance={:.3}",
                    avalanche.average_avalanche, avalanche.worst_bias, avalanche.variance
                );
                println!(
                    "    └─ Best bit {}: {:.3}%, Worst bit {}: {:.3}%",
                    best_bit.0, best_bit.1, worst_bit.0, worst_bit.1
                );
                println!("    └─ Distribution: Chi²={:.3}", distribution.chi_square);
                if let Some(collisions) = &collisions_opt {
                    if collisions.total_collisions > 0 {
                        match collisions.first_collision_at {
                            Some(pos) => {
                                println!(
                                "    └─ Integral ⨳: {} collisions, first at {} (expected at ~{})",
                                format_thousands(collisions.total_collisions),
                                format_thousands(pos),
                                format_thousands(collisions.expected_collision_at)
                            )
                            }
                            None => {
                                println!(
                                    "    └─ Integral ⨳: {} collisions",
                                    format_thousands(collisions.total_collisions)
                                )
                            }
                        }
                    } else {
                        println!(
                            "    └─ Integral ⨳: No collisions in {} samples",
                            format_thousands(collisions.total_tests)
                        );
                    }
                } else {
                    println!("    └─ Integral ⨳: Skipped (N-gram size > 8)");
                }
            } else {
                println!(" done");
            }

            let (total_collisions, collision_display, total_collisions_count) =
                if let Some(collisions) = &collisions_opt {
                    let display = match collisions.first_collision_at {
                        Some(pos) => format!("@{}", format_thousands(pos)),
                        None => "None".to_string(),
                    };
                    (
                        collisions.total_collisions,
                        display,
                        collisions.total_collisions,
                    )
                } else {
                    (0, "N/A".to_string(), 0)
                };

            // Calculate test duration
            let duration_ms = start_time.elapsed().as_millis() as u64;

            detailed_results.push(DetailedTestResult {
                hash_name: hash_func.name().to_string(),
                ngram_size: *ngram_size,
                ngrams_tested: actual_samples,
                avalanche_bias_display: format!("{:.5}%", avalanche.worst_bias),
                total_collisions,
                first_collision_display: collision_display,
                distribution_score_display: format!("{:.3}", distribution.chi_square),
                avalanche_bias: avalanche.worst_bias,
                distribution_score: distribution.chi_square,
                total_collisions_count,
                duration_ms,
            });
        }
    }

    // Aggregate results by hash function for main summary table
    let aggregated_results = aggregate_results(&detailed_results);

    // Print tabular results using tabled
    println!();
    println!("Hash Quality Analysis:");
    println!();

    let table = Table::new(&aggregated_results)
        .with(Style::psql())
        .modify(
            tabled::settings::object::Columns::new(1..6),
            Alignment::right(),
        )
        .to_string();
    println!("{}", table);

    // Export to CSV if requested
    if let Some(csv_path) = csv_output {
        match export_to_csv(&aggregated_results, csv_path) {
            Ok(()) => println!("\nResults exported to: {}", csv_path),
            Err(e) => eprintln!("Failed to export CSV: {}", e),
        }
    }

    // Leaders by n-gram size
    #[derive(Debug, Serialize, Deserialize, Tabled)]
    struct NGramSummary {
        #[tabled(rename = "N-gram")]
        ngram_size: usize,
        #[tabled(rename = "Lowest Bias")]
        lowest_bias_leaders: String,
        #[tabled(rename = "Highest Speed")]
        highest_speed_leaders: String,
        #[tabled(rename = "Lowest Collisions")]
        lowest_collisions_leaders: String,
    }

    let mut ngram_summaries = Vec::new();

    for ngram_size in ngram_sizes {
        let size_results: Vec<_> = detailed_results
            .iter()
            .filter(|r| r.ngram_size == *ngram_size)
            .collect();

        if !size_results.is_empty() {
            ngram_summaries.push(NGramSummary {
                ngram_size: *ngram_size,
                lowest_bias_leaders: find_bias_leaders(&size_results),
                highest_speed_leaders: find_speed_leaders(&size_results),
                lowest_collisions_leaders: find_collision_leaders(&size_results),
            });
        }
    }

    println!();
    println!("Leaders by N-gram size:");
    println!();
    let summary_table = Table::new(&ngram_summaries)
        .with(Style::psql())
        .modify(
            tabled::settings::object::Columns::new(2..5),
            Alignment::right(),
        )
        .to_string();
    println!("{}", summary_table);
}

/// Command line arguments
#[derive(Parser)]
#[command(name = "hashevals")]
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
    #[arg(long = "samples", default_value = "1000000")]
    samples: usize,

    /// N-gram sizes to test (comma-separated)
    #[arg(
        long = "ngrams",
        default_value = "1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20"
    )]
    ngram_sizes_str: String,

    /// Show detailed statistics for each individual test
    #[arg(long = "verbose", short = 'v')]
    verbose: bool,

    /// Export results to CSV file
    #[arg(long = "csv", value_name = "FILE")]
    csv_output: Option<String>,
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

    println!("HashEvals: Minimalistic Hash Function Quality Testing Suite");
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
    println!(
        "  Samples per size: {} n-grams",
        format_thousands(args.samples)
    );
    let total_tests = ngram_sizes
        .iter()
        .map(|&s| s * 8 * args.samples)
        .sum::<usize>();
    println!(
        "  Total avalanche tests: {} bit flips per hash function",
        format_thousands(total_tests)
    );

    test_hash_functions_tabular(
        &hash_functions,
        &ngram_sizes,
        args.samples,
        args.verbose,
        args.csv_output.as_deref(),
    );

    println!();
    println!("Notes:");
    println!("- Avalanche bias: <0.1% (excellent), <0.5% (very good), <1.0% (good)");
    println!("- Chi²: Lower values indicate better distribution uniformity");
    println!("- Integral ⨳: Collisions on random N-byte little-endian uints (only for N ≤ 8)");
    println!("  Expected collision ~√(π/2) × √samples by birthday paradox");
    println!("  Uses sample-sized bitset with hash % sample_size mapping");
}
