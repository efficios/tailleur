# Tailleur

A generic benchmark framework and runner.

This project aims to provide tools for blackbox benchmarking, with options to drop into other tools for microbenchmarking.

## Why

1. Existing benchmarking tools in Python such as ASV[1] and pytest-benchmark[2] are meant to benchmark Python projects.
2. Benchmarks should be able to return more metrics than just execution time.
3. Benchmark results should include more metadata on the running environment.

## References and more

[1]: https://github.com/airspeed-velocity/asv
[2]: https://github.com/ionelmc/pytest-benchmark
[3]: https://github.com/google/benchmark
[4]: https://github.com/sharkdp/hyperfine
