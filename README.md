# HashEvals

HashEvals is a Rust program that stress-tests hash functions for avalanche quality, integral collisions, and distribution skew across variable-length n-grams.
The suite draws inspiration from the SMHasher family of benchmarks while keeping a lean codebase that is easy to extend with new hash primitives.

What It Tests?

- __Avalanche behavior__ – flips every input bit and measures how many output bits change, tracking worst-case bias and per-bit variance. Ideally, flipping any input bit changes each output bit with 50% probability.
- __Integer collisions__ – hashes random integers with only last `n` bytes populated and watches for birthday-paradox collisions (for `n ≤ 8`). Representative of open-addressing hash tables with integer keys.
- __Distribution probe__ – runs large bucketed Chi² checks to highlight skewed output distributions. Representative of constructing bucketed hash tables or load balancers.

Each test operates over continuous random buffers generated with `ChaCha20Rng` so results are deterministic under the same input `--seed`.
Hashes returning 32-bit values (e.g. `Crc32`, `RabinKarp32`) still participate in the avalanche and distribution checks.

## Results

```sh
 Function    |   Avg.Bias | Worst.Bias | Integral ⨳ |     Chi² |    Throughput 
-------------+------------+------------+------------+----------+---------------
 Blake3      |  0.19546 % |  3.75977 % |   41.107 % | 1814.833 |   600.5 MiB/s 
 SeaHash     |  0.22700 % |  4.44336 % |   40.740 % | 1787.435 |  3237.3 MiB/s 
 FoldHash    |  0.23887 % |  4.88281 % |   41.105 % | 1756.197 | 10507.5 MiB/s 
 SipHash     |  0.23944 % |  4.88281 % |   40.435 % | 1793.684 |  2735.3 MiB/s 
 FarmHash    |  0.24465 % |  5.07812 % |   40.809 % | 1769.331 |  6201.4 MiB/s 
 xxHash3     |  0.24745 % |  5.07812 % |   40.854 % | 1800.470 |  8182.9 MiB/s 
 aHash       |  0.25809 % |  5.37109 % |   40.999 % | 1799.570 | 11604.1 MiB/s 
 gxHash      |  0.26063 % |  5.41992 % |   40.616 % | 1783.136 | 14838.3 MiB/s 
 StringZilla |  0.26147 % |  5.51758 % |   40.677 % | 1787.186 | 10539.1 MiB/s 
 MurMur3     |  0.26568 % |  5.56641 % |   40.924 % | 1766.211 |  4080.0 MiB/s 
 FxHash      |  1.88428 % |  6.90654 % |   40.860 % | 1800.042 | 11028.6 MiB/s 
 Crc32       | 15.87577 % | 37.50000 % |   41.672 % | 1299.878 |  2925.2 MiB/s 
 RabinKarp32 | 50.00000 % | 50.00000 % |   38.378 % | 1317.204 |   334.7 MiB/s
```

### Tiny N-grams (≤ 8 bytes):

```sh
 Function    |   Avg.Bias | Worst.Bias | Integral ⨳ |     Chi² |   Throughput 
-------------+------------+------------+------------+----------+--------------
 Blake3      |  0.54203 % |  3.75977 % |   41.107 % | 3670.615 |   71.0 MiB/s 
 SeaHash     |  0.63433 % |  4.44336 % |   40.740 % | 3584.745 |  770.4 MiB/s 
 SipHash     |  0.68561 % |  4.88281 % |   40.435 % | 3613.683 |  498.6 MiB/s 
 FoldHash    |  0.69112 % |  4.88281 % |   41.105 % | 3546.575 | 1533.8 MiB/s 
 FarmHash    |  0.70883 % |  5.07812 % |   40.809 % | 3516.119 |  977.3 MiB/s 
 xxHash3     |  0.71570 % |  5.07812 % |   40.854 % | 3646.251 | 1346.6 MiB/s 
 aHash       |  0.74339 % |  5.37109 % |   40.999 % | 3691.414 | 1208.6 MiB/s 
 StringZilla |  0.75514 % |  5.51758 % |   40.677 % | 3578.100 | 1099.8 MiB/s 
 gxHash      |  0.75618 % |  5.41992 % |   40.616 % | 3551.557 | 1576.2 MiB/s 
 MurMur3     |  0.77171 % |  5.56641 % |   40.924 % | 3563.656 |  647.8 MiB/s 
 FxHash      |  4.33077 % |  6.90654 % |   40.860 % | 3637.124 | 1506.8 MiB/s 
 Crc32       | 19.01042 % | 37.50000 % |   41.672 % | 1969.218 |  715.6 MiB/s 
 RabinKarp32 | 50.00000 % | 50.00000 % |   38.378 % | 2014.422 |  623.0 MiB/s
 ```

### Short N-grams (9-32 bytes):

```sh
 Function    |   Avg.Bias | Worst.Bias |     Chi² |   Throughput 
-------------+------------+------------+----------+--------------
 FoldHash    |  0.04853 % |  0.05767 % | 1021.043 | 3913.9 MiB/s 
 xxHash3     |  0.04955 % |  0.06000 % | 1024.132 | 3921.6 MiB/s 
 Blake3      |  0.05026 % |  0.07045 % | 1018.744 |  209.8 MiB/s 
 FarmHash    |  0.05051 % |  0.07125 % | 1035.789 | 2611.0 MiB/s 
 SipHash     |  0.05102 % |  0.06377 % | 1035.855 | 1254.5 MiB/s 
 MurMur3     |  0.05212 % |  0.07022 % | 1009.473 | 1667.1 MiB/s 
 gxHash      |  0.05277 % |  0.07558 % | 1043.073 | 4497.0 MiB/s 
 StringZilla |  0.05500 % |  0.07538 % | 1033.640 | 3127.5 MiB/s 
 aHash       |  0.05623 % |  0.06669 % | 1006.648 | 3860.8 MiB/s 
 SeaHash     |  0.05627 % |  0.07600 % | 1028.949 | 1772.4 MiB/s 
 FxHash      |  1.22235 % |  5.18011 % | 1019.290 | 4265.9 MiB/s 
 Crc32       | 14.78365 % | 20.31250 % | 1008.800 |  822.2 MiB/s 
 RabinKarp32 | 50.00000 % | 50.00000 % | 1037.807 |  492.0 MiB/s 
```

### Long N-grams (> 32 bytes):

```sh
 Function    |   Avg.Bias | Worst.Bias |     Chi² |    Throughput 
-------------+------------+------------+----------+---------------
 FarmHash    |  0.04636 % |  0.05584 % | 1029.621 |  8857.2 MiB/s 
 Blake3      |  0.04798 % |  0.05692 % | 1065.318 |  1019.4 MiB/s 
 FoldHash    |  0.04826 % |  0.06458 % |  961.860 | 16206.8 MiB/s 
 aHash       |  0.04836 % |  0.06597 % |  995.109 | 21306.8 MiB/s 
 gxHash      |  0.05027 % |  0.06848 % | 1028.712 | 29073.2 MiB/s 
 StringZilla |  0.05059 % |  0.05597 % | 1031.981 | 21188.3 MiB/s 
 xxHash3     |  0.05189 % |  0.06267 % | 1021.496 | 10999.5 MiB/s 
 SipHash     |  0.05278 % |  0.06345 % | 1008.981 |  3671.8 MiB/s 
 MurMur3     |  0.05368 % |  0.06795 % | 1009.214 |  5897.9 MiB/s 
 SeaHash     |  0.05380 % |  0.07005 % | 1034.406 |  3947.8 MiB/s 
 FxHash      |  0.05646 % |  0.06623 % | 1042.229 | 16945.9 MiB/s 
 Crc32       | 14.06250 % | 18.75000 % | 1038.094 |  4906.4 MiB/s 
 RabinKarp32 | 50.00000 % | 50.00000 % |  992.942 |   318.1 MiB/s
```

## Replicating the Results

Build and execute like any standard Cargo binary:

```sh
cargo run --release -- --list-hashes  # list supported hash functions
cargo run --release -- --help         # show CLI options
```

To run a very small sample set for a quick sanity check:

```sh
RUSTFLAGS="-C target-cpu=native" cargo run --release -- --samples 100 --verbose
```

For a proper comparison, consider running for 1 million samples:

```sh
RUSTFLAGS="-C target-cpu=native" cargo run --release -- --samples 1000000
```

## Contributing

Add new hashers by implementing `HashFunction` trait in `hash_functions.rs` and pushing the boxed instance into `get_all_hash_functions()`.
Each implementation exposes its display name, bit width, and `hash(&[u8]) -> u64` method, which the test harness dispatches automatically across all metrics.
If you want to add a new stress-testing methodology, please open an issue or PR to discuss the design!
