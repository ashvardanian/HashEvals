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
 Blake3      |  0.15142 % |  3.75977 % |   42.347 % | 2021.527 |   582.1 MiB/s 
 SeaHash     |  0.17826 % |  4.44336 % |   42.055 % | 2012.333 |  3525.7 MiB/s 
 SipHash     |  0.19405 % |  4.88281 % |   41.682 % | 2010.550 |  2734.0 MiB/s 
 FoldHash    |  0.19693 % |  4.88281 % |   42.379 % | 2022.245 | 10712.6 MiB/s 
 FarmHash    |  0.19945 % |  5.07812 % |   42.112 % | 1985.036 |  6123.6 MiB/s 
 xxHash3     |  0.20226 % |  5.07812 % |   42.096 % | 2006.815 |  8122.3 MiB/s 
 gxHash      |  0.21399 % |  5.41992 % |   41.964 % | 1988.415 |  1020.3 MiB/s 
 StringZilla |  0.21524 % |  5.51758 % |   41.932 % | 1996.037 | 10994.1 MiB/s 
 MurMur3     |  0.21968 % |  5.56641 % |   42.200 % | 1993.416 |  3914.6 MiB/s 
 aHash       |  0.24295 % |  6.20117 % |   42.094 % | 1988.110 | 11371.3 MiB/s 
 FxHash      |  1.86375 % |  6.92404 % |   42.130 % | 2022.595 | 11154.7 MiB/s 
 Crc32       | 15.87577 % | 37.50000 % |   42.828 % | 1315.053 |  2811.4 MiB/s 
 RabinKarp32 | 50.00000 % | 50.00000 % |   38.630 % | 1318.352 |   310.4 MiB/s 
```

```
Configuration:
  Hash functions: StringZilla, SipHash, aHash, xxHash3, gxHash, Crc32, MurMur3, FarmHash, Blake3, FxHash, FoldHash, SeaHash, RabinKarp32
  N-gram sizes: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 30, 60, 100, 200, 300, 400, 500] bytes
  Samples per size: 10'000'000 n-grams
  Random seed: 42
  Total avalanche tests: 15'040'000'000 bit flips per hash function
  Optimization: For N > 8, randomly sample 64 unique bit positions to avoid quadratic complexity
```

### Tiny N-grams (≤ 8 bytes):

```sh
 Function    |   Avg.Bias | Worst.Bias | Integral ⨳ |     Chi² |   Throughput 
-------------+------------+------------+------------+----------+--------------
 Blake3      |  0.49842 % |  3.75977 % |   42.347 % | 4401.729 |   66.1 MiB/s 
 SeaHash     |  0.59017 % |  4.44336 % |   42.055 % | 4353.642 |  791.8 MiB/s 
 SipHash     |  0.64224 % |  4.88281 % |   41.682 % | 4330.098 |  479.7 MiB/s 
 FoldHash    |  0.65240 % |  4.88281 % |   42.379 % | 4411.870 | 1533.8 MiB/s 
 FarmHash    |  0.66107 % |  5.07812 % |   42.112 % | 4321.060 |  890.7 MiB/s 
 xxHash3     |  0.67025 % |  5.07812 % |   42.096 % | 4387.052 | 1373.7 MiB/s 
 gxHash      |  0.70988 % |  5.41992 % |   41.964 % | 4302.142 | 1500.2 MiB/s 
 StringZilla |  0.71370 % |  5.51758 % |   41.932 % | 4305.659 | 1055.7 MiB/s 
 MurMur3     |  0.72892 % |  5.56641 % |   42.200 % | 4314.708 |  614.1 MiB/s 
 aHash       |  0.80697 % |  6.20117 % |   42.094 % | 4275.918 | 1320.6 MiB/s 
 FxHash      |  4.33818 % |  6.92404 % |   42.130 % | 4384.974 | 1652.4 MiB/s 
 Crc32       | 19.01042 % | 37.50000 % |   42.828 % | 1994.412 |  563.5 MiB/s 
 RabinKarp32 | 50.00000 % | 50.00000 % |   38.630 % | 1981.562 |  574.9 MiB/s 
 ```

### Short N-grams (9-32 bytes):

```sh
 Function    |   Avg.Bias | Worst.Bias |     Chi² |   Throughput 
-------------+------------+------------+----------+--------------
 SeaHash     |  0.00469 % |  0.00545 % | 1034.607 | 1893.7 MiB/s 
 xxHash3     |  0.00500 % |  0.00627 % | 1012.462 | 4016.5 MiB/s 
 FarmHash    |  0.00502 % |  0.00645 % | 1010.069 | 2688.5 MiB/s 
 FoldHash    |  0.00506 % |  0.00715 % | 1013.351 | 4202.9 MiB/s 
 SipHash     |  0.00520 % |  0.00638 % | 1044.576 | 1272.2 MiB/s 
 gxHash      |  0.00523 % |  0.00723 % | 1011.157 | 4359.3 MiB/s 
 Blake3      |  0.00536 % |  0.00818 % | 1012.602 |  214.6 MiB/s 
 MurMur3     |  0.00544 % |  0.00682 % | 1017.549 | 1646.4 MiB/s 
 StringZilla |  0.00556 % |  0.00730 % | 1010.320 | 3470.7 MiB/s 
 aHash       |  0.00576 % |  0.00676 % | 1018.854 | 3913.9 MiB/s 
 FxHash      |  1.19828 % |  5.17577 % | 1017.994 | 4507.2 MiB/s 
 Crc32       | 14.78365 % | 20.31250 % | 1038.868 |  807.9 MiB/s 
 RabinKarp32 | 50.00000 % | 50.00000 % | 1033.444 |  497.3 MiB/s 
```

### Long N-grams (> 32 bytes):

```sh
 Function    |   Avg.Bias | Worst.Bias |     Chi² |    Throughput 
-------------+------------+------------+----------+---------------
 aHash       |  0.00485 % |  0.00592 % | 1037.755 | 19606.7 MiB/s 
 MurMur3     |  0.00491 % |  0.00545 % | 1012.738 |  5598.8 MiB/s 
 StringZilla |  0.00495 % |  0.00589 % | 1052.259 | 21763.4 MiB/s 
 gxHash      |  0.00512 % |  0.00605 % | 1020.838 |   921.4 MiB/s 
 SeaHash     |  0.00513 % |  0.00626 % | 1008.996 |  4353.9 MiB/s 
 FarmHash    |  0.00523 % |  0.00633 % |  982.769 |  8735.3 MiB/s 
 Blake3      |  0.00523 % |  0.00584 % | 1033.929 |   969.0 MiB/s 
 FoldHash    |  0.00538 % |  0.00718 % | 1022.016 | 16155.2 MiB/s 
 SipHash     |  0.00563 % |  0.00687 % | 1010.762 |  3673.6 MiB/s 
 xxHash3     |  0.00569 % |  0.00692 % |  987.596 | 10743.6 MiB/s 
 FxHash      |  0.00635 % |  0.01340 % | 1049.392 | 16451.8 MiB/s 
 Crc32       | 14.06250 % | 18.75000 % | 1007.642 |  4796.7 MiB/s 
 RabinKarp32 | 50.00000 % | 50.00000 % | 1051.375 |   293.0 MiB/s 
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
