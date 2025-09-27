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
 Blake3      |  0.19546 % |  3.75977 % |   41.107 % | 1814.833 |   600.0 MiB/s 
 aHash       |  0.20945 % |  4.10156 % |   40.740 % | 1794.325 | 12234.4 MiB/s 
 SeaHash     |  0.22700 % |  4.44336 % |   41.026 % | 1787.435 |  3559.3 MiB/s 
 FoldHash    |  0.23887 % |  4.88281 % |   40.435 % | 1756.197 | 11269.1 MiB/s 
 SipHash     |  0.23944 % |  4.88281 % |   41.105 % | 1793.684 |  2819.1 MiB/s 
 FarmHash    |  0.24465 % |  5.07812 % |   40.809 % | 1769.331 |  6449.6 MiB/s 
 xxHash3     |  0.24745 % |  5.07812 % |   40.854 % | 1800.470 |  8727.6 MiB/s 
 gxHash      |  0.26063 % |  5.41992 % |   40.677 % | 1783.136 | 14775.9 MiB/s 
 StringZilla |  0.26147 % |  5.51758 % |   40.616 % | 1787.186 | 10451.3 MiB/s 
 MurMur3     |  0.26568 % |  5.56641 % |   40.924 % | 1766.211 |  4224.2 MiB/s 
 FxHash      |  1.88428 % |  6.90654 % |   40.860 % | 1800.042 | 11704.6 MiB/s 
 Crc32       | 15.87577 % | 37.50000 % |   41.672 % | 1299.878 |  3017.1 MiB/s 
 RabinKarp32 | 50.00000 % | 50.00000 % |   38.378 % | 1317.204 |   250.1 MiB/s 
```

### Tiny N-grams (≤ 8 bytes):

```sh
 Function    |   Avg.Bias | Worst.Bias | Integral ⨳ |     Chi² |   Throughput 
-------------+------------+------------+------------+----------+--------------
 Blake3      |  0.54203 % |  3.75977 % |   41.107 % | 3670.615 |   69.2 MiB/s 
 SeaHash     |  0.63433 % |  4.44336 % |   40.740 % | 3584.745 |  751.7 MiB/s 
 aHash       |  0.65054 % |  4.63867 % |   41.026 % | 3689.559 | 1248.3 MiB/s 
 SipHash     |  0.68561 % |  4.88281 % |   40.435 % | 3613.683 |  468.5 MiB/s 
 FoldHash    |  0.69112 % |  4.88281 % |   41.105 % | 3546.575 | 1461.7 MiB/s 
 FarmHash    |  0.70883 % |  5.07812 % |   40.809 % | 3516.119 |  909.7 MiB/s 
 xxHash3     |  0.71570 % |  5.07812 % |   40.854 % | 3646.251 | 1300.5 MiB/s 
 StringZilla |  0.75514 % |  5.51758 % |   40.677 % | 3578.100 |  971.7 MiB/s 
 gxHash      |  0.75618 % |  5.41992 % |   40.616 % | 3551.557 | 1500.2 MiB/s 
 MurMur3     |  0.77171 % |  5.56641 % |   40.924 % | 3563.656 |  606.5 MiB/s 
 FxHash      |  4.33077 % |  6.90654 % |   40.860 % | 3637.124 | 1540.7 MiB/s 
 Crc32       | 19.01042 % | 37.50000 % |   41.672 % | 1969.218 |  685.5 MiB/s 
 RabinKarp32 | 50.00000 % | 50.00000 % |   38.378 % | 2014.422 |  577.8 MiB/s
```

### Short N-grams (9-32 bytes):

```sh
 Function    |   Avg.Bias | Worst.Bias |     Chi² |   Throughput 
-------------+------------+------------+----------+--------------
 FoldHash    |  0.04853 % |  0.05767 % | 1021.043 | 4107.6 MiB/s 
 xxHash3     |  0.04955 % |  0.06000 % | 1024.132 | 4579.7 MiB/s 
 Blake3      |  0.05026 % |  0.07045 % | 1018.744 |  212.1 MiB/s 
 FarmHash    |  0.05051 % |  0.07125 % | 1035.789 | 2763.1 MiB/s 
 aHash       |  0.05070 % |  0.07447 % | 1026.718 | 4185.3 MiB/s 
 SipHash     |  0.05102 % |  0.06377 % | 1035.855 | 1256.9 MiB/s 
 MurMur3     |  0.05212 % |  0.07022 % | 1009.473 | 1694.0 MiB/s 
 gxHash      |  0.05277 % |  0.07558 % | 1043.073 | 4517.4 MiB/s 
 StringZilla |  0.05500 % |  0.07538 % | 1033.640 | 2899.8 MiB/s 
 SeaHash     |  0.05627 % |  0.07600 % | 1028.949 | 1888.3 MiB/s 
 FxHash      |  1.22235 % |  5.18011 % | 1019.290 | 4527.7 MiB/s 
 Crc32       | 14.78365 % | 20.31250 % | 1008.800 |  847.7 MiB/s 
 RabinKarp32 | 50.00000 % | 50.00000 % | 1037.807 |  498.9 MiB/s 
```

### Long N-grams (> 32 bytes):

```sh
 Function    |   Avg.Bias | Worst.Bias |     Chi² |    Throughput 
-------------+------------+------------+----------+---------------
 FarmHash    |  0.04636 % |  0.05584 % | 1029.621 |  9238.6 MiB/s 
 aHash       |  0.04735 % |  0.05119 % | 1028.733 | 21578.4 MiB/s 
 Blake3      |  0.04798 % |  0.05692 % | 1065.318 |  1020.0 MiB/s 
 FoldHash    |  0.04826 % |  0.06458 % |  961.860 | 17880.7 MiB/s 
 gxHash      |  0.05027 % |  0.06848 % | 1028.712 | 28475.5 MiB/s 
 StringZilla |  0.05059 % |  0.05597 % | 1031.981 | 23365.6 MiB/s 
 xxHash3     |  0.05189 % |  0.06267 % | 1021.496 | 11541.2 MiB/s 
 SipHash     |  0.05278 % |  0.06345 % | 1008.981 |  3830.6 MiB/s 
 MurMur3     |  0.05368 % |  0.06795 % | 1009.214 |  6187.8 MiB/s 
 SeaHash     |  0.05380 % |  0.07005 % | 1034.406 |  4391.6 MiB/s 
 FxHash      |  0.05646 % |  0.06623 % | 1042.229 | 17859.8 MiB/s 
 Crc32       | 14.06250 % | 18.75000 % | 1038.094 |  5084.9 MiB/s 
 RabinKarp32 | 50.00000 % | 50.00000 % |  992.942 |   231.9 MiB/s
```

## Replicating the Results

Build and execute like any standard Cargo binary:

```sh
cargo run --release -- --list-hashes  # list supported hash functions
cargo run --release -- --help         # show CLI options
RUSTFLAGS="-C target-cpu=native" \
  cargo run --release \
  --samples 100 --verbose             # run the full benchmark suite
```

## Contributing

Add new hashers by implementing `HashFunction` trait in `hash_functions.rs` and pushing the boxed instance into `get_all_hash_functions()`.
Each implementation exposes its display name, bit width, and `hash(&[u8]) -> u64` method, which the test harness dispatches automatically across all metrics.
If you want to add a new stress-testing methodology, please open an issue or PR to discuss the design!
