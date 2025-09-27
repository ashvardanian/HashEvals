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

/// Perform rigorous avalanche analysis using n-grams from continuous buffer
pub fn test_avalanche<H, F>(
    hash_fn: F,
    ngram_length: usize,
    sample_count: usize,
) -> AvalancheResults
where
    H: HashValue,
    F: Fn(&[u8]) -> H,
{
    // Generate a large buffer and test n-grams
    let buffer_size = ngram_length + sample_count; // Ensure we have enough n-grams
    let buffer = generate_test_buffer(buffer_size, 42);
    let output_bits = H::total_bits() as usize;

    let mut bit_flip_counts = vec![0usize; output_bits];
    let mut total_tests = 0;

    // Pre-allocate reusable buffer for modified n-grams
    let mut modified_buffer = vec![0u8; ngram_length];

    // Use iterator to process n-grams without copying
    for (idx, ngram) in NGramIterator::new(&buffer, ngram_length).enumerate() {
        if idx >= sample_count {
            break;
        }

        let original_hash = hash_fn(ngram);
        let total_input_bits = ngram_length * 8;

        // Test flipping each input bit
        for bit_pos in 0..total_input_bits {
            let byte_idx = bit_pos / 8;
            let bit_idx = bit_pos % 8;

            // Copy to reusable buffer and flip bit
            modified_buffer.copy_from_slice(ngram);
            modified_buffer[byte_idx] ^= 1 << bit_idx;

            let modified_hash = hash_fn(&modified_buffer);
            let xor_result = original_hash.xor(modified_hash);

            // Count which output bits flipped
            for output_bit in 0..output_bits {
                if (xor_result.to_u64() >> output_bit) & 1 == 1 {
                    bit_flip_counts[output_bit] += 1;
                }
            }

            total_tests += 1;
        }
    }

    // Calculate per-bit statistics
    let per_bit_scores: Vec<f64> = bit_flip_counts
        .iter()
        .map(|&count| (count as f64 / total_tests as f64) * 100.0)
        .collect();

    let average_avalanche = per_bit_scores.iter().sum::<f64>() / output_bits as f64;

    // Calculate bias (deviation from ideal 50%)
    let biases: Vec<f64> = per_bit_scores
        .iter()
        .map(|&score| (score - 50.0).abs())
        .collect();

    let worst_bias = biases.iter().fold(0.0f64, |a, &b| a.max(b));
    let best_bias = biases.iter().fold(f64::INFINITY, |a, &b| a.min(b));

    // Calculate variance
    let mean = average_avalanche;
    let variance = per_bit_scores
        .iter()
        .map(|score| (score - mean).powi(2))
        .sum::<f64>()
        / output_bits as f64;

    AvalancheResults {
        average_avalanche,
        worst_bias,
        best_bias,
        variance,
        per_bit_scores,
        total_tests,
    }
}

/// Test differential patterns using n-grams
pub fn test_differential<H, F>(
    hash_fn: F,
    ngram_length: usize,
    diff_bits: usize,
    sample_count: usize,
) -> DifferentialResults
where
    H: HashValue,
    F: Fn(&[u8]) -> H,
{
    // Generate buffer and pre-allocate modified buffer
    let buffer_size = ngram_length + sample_count;
    let buffer = generate_test_buffer(buffer_size, 123);
    let mut modified_buffer = vec![0u8; ngram_length];
    let mut collision_count = 0;

    // Test differential patterns on n-grams
    for (idx, ngram) in NGramIterator::new(&buffer, ngram_length).enumerate() {
        if idx >= sample_count {
            break;
        }

        let hash1 = hash_fn(ngram);

        // Create a variant with exactly diff_bits flipped
        modified_buffer.copy_from_slice(ngram);

        // For simplicity, flip the first diff_bits bits
        for bit_idx in 0..diff_bits {
            let byte_idx = bit_idx / 8;
            let bit_pos = bit_idx % 8;
            if byte_idx < ngram_length {
                modified_buffer[byte_idx] ^= 1 << bit_pos;
            }
        }

        let hash2 = hash_fn(&modified_buffer);

        if hash1 == hash2 {
            collision_count += 1;
        }
    }

    // Calculate expected collision rate
    let hash_space_size = 2.0_f64.powi(H::total_bits() as i32);
    let expected_collisions = (sample_count as f64) / hash_space_size;
    let collision_ratio = collision_count as f64 / expected_collisions.max(1e-10);

    DifferentialResults {
        expected_collisions,
        actual_collisions: collision_count,
        collision_ratio,
        worst_pattern: None, // TODO: Track which bit patterns cause most collisions
        total_tests: sample_count,
    }
}

/// Test distribution uniformity using n-grams
pub fn test_distribution<H, F>(
    hash_fn: F,
    ngram_length: usize,
    sample_count: usize,
    bucket_count: usize,
) -> DistributionResults
where
    H: HashValue,
    F: Fn(&[u8]) -> H,
{
    // Generate buffer for n-grams
    let buffer_size = ngram_length + sample_count;
    let buffer = generate_test_buffer(buffer_size, 789);
    let mut buckets = vec![0usize; bucket_count];

    // Hash n-grams and distribute into buckets
    for (idx, ngram) in NGramIterator::new(&buffer, ngram_length).enumerate() {
        if idx >= sample_count {
            break;
        }

        let hash = hash_fn(ngram);
        let bucket = (hash.to_u64() % bucket_count as u64) as usize;
        buckets[bucket] += 1;
    }

    // Calculate chi-square statistic
    let expected = sample_count as f64 / bucket_count as f64;
    let chi_square: f64 = buckets
        .iter()
        .map(|&observed| {
            let diff = observed as f64 - expected;
            (diff * diff) / expected
        })
        .sum();

    // Simplified p-value calculation (for large samples, chi-square approaches normal)
    let degrees_of_freedom = bucket_count - 1;
    let p_value = 1.0 - (chi_square / degrees_of_freedom as f64).min(1.0);

    // Uniformity score (0-100, higher is better)
    let max_deviation = buckets
        .iter()
        .map(|&count| ((count as f64 - expected) / expected).abs())
        .fold(0.0f64, |a, b| a.max(b));

    let uniformity_score = ((1.0 - max_deviation.min(1.0)) * 100.0).max(0.0);

    DistributionResults {
        chi_square,
        p_value,
        uniformity_score,
        bucket_count,
    }
}

/// Test hash functions across multiple n-gram sizes and print tabular results
fn test_hash_functions_tabular(
    hash_functions: &[Box<dyn HashFunction>],
    ngram_sizes: &[usize],
    samples_per_size: usize,
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

    for ngram_size in ngram_sizes {
        println!(
            "\nTesting {}-grams with {} samples each...",
            ngram_size, samples_per_size
        );

        for hash_func in hash_functions {
            print!("  Testing {}...", hash_func.name());

            // Run tests based on bit width
            let (avalanche, differential, distribution, actual_samples) = if hash_func.bits() == 64
            {
                let aval = test_avalanche(|d| hash_func.hash(d), *ngram_size, samples_per_size);
                let diff =
                    test_differential(|d| hash_func.hash(d), *ngram_size, 1, samples_per_size);
                let dist = test_distribution(
                    |d| hash_func.hash(d),
                    *ngram_size,
                    samples_per_size * 10,
                    1024,
                );
                let samples = samples_per_size.min(dist.bucket_count * 10); // Actual samples tested
                (aval, diff, dist, samples)
            } else {
                let aval =
                    test_avalanche(|d| hash_func.hash(d) as u32, *ngram_size, samples_per_size);
                let diff = test_differential(
                    |d| hash_func.hash(d) as u32,
                    *ngram_size,
                    1,
                    samples_per_size,
                );
                let dist = test_distribution(
                    |d| hash_func.hash(d) as u32,
                    *ngram_size,
                    samples_per_size * 10,
                    1024,
                );
                let samples = samples_per_size.min(dist.bucket_count * 10);
                (aval, diff, dist, samples)
            };

            all_results.push(TestResult {
                hash_name: hash_func.name().to_string(),
                ngram_size: *ngram_size,
                avalanche_bias: avalanche.worst_bias,
                differential_collisions: differential.actual_collisions,
                distribution_score: distribution.uniformity_score,
                ngrams_tested: actual_samples,
            });

            println!(" done");
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

    test_hash_functions_tabular(&hash_functions, &ngram_sizes, args.samples);

    println!("\nNotes:");
    println!("- Avalanche bias: <0.1% (excellent), <0.5% (very good), <1.0% (good)");
    println!("- Distribution score: Higher is better (100% = perfect uniformity)");
    println!("- Differential collisions: Should be 0 or very close to 0");
    println!("- CRC32 is expected to have poor avalanche (not designed for it)");
}
