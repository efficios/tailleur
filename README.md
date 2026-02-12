# Tailleur

A generic benchmark framework and runner.

This project aims to provide tools for blackbox benchmarking, with options to drop into other tools for microbenchmarking.

## Running

From source:

    git clone https://github.com/kienanstewart/tailleur.git && cd tailleur
    poetry install
    poetry run tailleur

## Configuration

### Defaults

A JSON or YAML file

    ---
    # config.yaml
    config:
      runs: 10   # The number of runs for each benchmark
    search_paths:
      - /path/to/x
      - relative/path/to/x

Set the defaults via command-line:


    poetry run tailleur --config /path/to/config.yaml

### Benchmark selection


#### via configuration file

A JSON or YAML file

    ---
    # suite.yaml
    search_paths:
      - /path/to/x
      - relative_to_cwd/x

    benchmarks:
      - name: [module.]ClassName
        # Configuration is merged with the defaults
        config:
          runs: 2
        params:
          - # set1
            param_X: vX
            param_Y: vY
          - # set2
            param_X: vX_2
            param_Y: vY_2

Specify it as follows:

    poetry run tailleur --benchmarks suite.yaml

## Why

1. Existing benchmarking tools in Python such as ASV[1] and pytest-benchmark[2] are meant to benchmark Python projects.
2. Benchmarks should be able to return more metrics than just execution time.
3. Benchmark results should include more metadata on the running environment.

## References and more

[1]: https://github.com/airspeed-velocity/asv
[2]: https://github.com/ionelmc/pytest-benchmark
[3]: https://github.com/google/benchmark
[4]: https://github.com/sharkdp/hyperfine
