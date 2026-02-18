#!/usr/bin/env python3
#
# SPDX-License-Identifier: GPL-2.0-only
# SPDX-FileCopyrightText: 2026 Kienan Stewart <kstewart@efficios.com>
#

import argparse
import enum
import importlib
import json
import logging
import os
import pathlib
import pkgutil
import platform
import re
import sys

import yaml


class FilterModule(enum.StrEnum):
    NAME = "name"
    RE = "re"


def discover_benchmarks(search_dirs):
    benchmarks = list()
    for finder, name, is_pkg in pkgutil.iter_modules(search_dirs):
        try:
            sys.path.insert(0, finder.path)
            mod = importlib.import_module(name)
            for k, v in mod.__dict__.items():
                if isinstance(v, type):
                    logging.info(
                        "Found benchmark {}.{}".format(mod.__name__, v.__name__)
                    )
                    benchmarks.append(
                        {
                            "name": "{}.{}".format(mod.__name__, v.__name__),
                            "cls": v,
                            "config": dict(),
                            "parameters": list(),
                        }
                    )
            sys.path = sys.path[1:]
        except Exception as e:
            logging.error("Failed to import module {}: {}".format(name, e))

    return benchmarks


def find_benchmark_by_name(all_benchmarks, name):
    options = list()
    for x in all_benchmarks:
        if x["name"] == name or (
            name.find(".") == -1 and x["name"].rfind(".{}".format(name)) != -1
        ):
            options.append(x)

    if len(options) > 2:
        raise Exception(
            "Multiple ({}) matches for benchmark by name '{}'".format(
                len(options), name
            )
        )

    if len(options) == 0:
        raise Exception("No match for benchmark by name '{}'".format(name))

    return options[0]["cls"]


def filter_benchmarks(benchmarks, benchmark_filters=list()):
    kept = list()
    keep_default = len(benchmark_filters) == 0
    logging.debug("Keep benchmarks by default: {}".format(keep_default))
    for benchmark in benchmarks:
        keep = keep_default
        for benchmark_filter in benchmark_filters:
            invert = False
            if benchmark_filter[0] == "!":
                invert = True
                benchmark_filter = benchmark_filter[1:]

            filter_module = FilterModule.NAME
            if benchmark_filter.startswith("{}:".format(FilterModule.RE)):
                filter_module = FilterModule.RE
                benchmark_filter = benchmark_filter.split(":")[1]
            elif benchmark_filter.startswith("{}:".format(FilterModule.NAME)):
                filter_module = FilterModule.NAME
                benchmark_filter = benchmark_filter.split(":")[1]
            elif benchmark_filter.rfind(":") != -1:
                raise Exception(
                    "Unknown filter module '{}' in filter '{}'".format(
                        benchmark_filter.split(":")[0], benchmark_filter
                    )
                )

            if filter_module == FilterModule.NAME:
                if benchmark_filter == benchmark["name"] or (
                    benchmark_filter.find(".") == -1
                    and benchmark["name"].rfind(".{}".format(benchmark_filter)) != -1
                ):
                    logging.debug(
                        "Benchmark '{}' matches filter '{}', invert={}".format(
                            benchmark["name"], benchmark_filter, invert
                        )
                    )
                    keep = True
            elif filter_module == FilterModule.RE:
                if re.match(benchmark_filter, benchmark["name"]):
                    keep = True

            if invert:
                keep = not keep

        if keep:
            kept.append(benchmark)

    return kept


def run_benchmarks(config=dict(), suite=dict(), benchmark_filters=list()):
    # Step 1: Load from search dirs and then transform into a set of benchmarks to run
    paths = config["search_paths"]
    if "search_paths" in suite:
        paths.extend(suite["search_paths"])

    all_benchmarks = discover_benchmarks(paths)
    config["search_paths"] = [str(x) for x in paths]
    if "runs" not in config:
        config["runs"] = 10

    if "benchmarks" in suite:
        benchmarks = list()
        # Check if it exists in all_benchmarks
        for x in suite["benchmarks"]:
            try:
                benchmark = {
                    "cls": find_benchmark_by_name(all_benchmarks, x["name"]),
                    "config": config | x.get("config", dict()),
                    "parameters": x.get("parameters", list()),
                }
                benchmark["name"] = "{}.{}".format(
                    benchmark["cls"].__module__, benchmark["cls"].__name__
                )
                benchmarks.append(benchmark)
            except Exception as e:
                logging.warning(
                    "Error finding benchmark by name '{}': {}".format(
                        x.get("name", "<no name>"), str(e)
                    )
                )
    else:
        benchmarks = all_benchmarks

    benchmarks = filter_benchmarks(benchmarks, benchmark_filters)
    results = {
        "config": config,
        "metadata": get_generic_metadata(),
        "results": list(),
    }
    for benchmark in benchmarks:
        result = run_benchmark(
            benchmark["cls"],
            config | benchmark.get("config", dict()),
            benchmark.get("parameters", list()),
        )
        if type(result) is list:
            # A benchmark that performed more than one parameter set may return a list
            results["results"].extend(result)
        else:
            results["results"].append(result)

    return results


def run_benchmark(cls, config=dict(), parameter_sets=list()):
    metrics = cls.metrics()
    metadata = cls.metadata()
    all_results = list()
    if len(parameter_sets) == 0:
        parameter_sets = cls.default_parameter_sets()

    for index, parameter_set in enumerate(parameter_sets):
        logging.info(
            "Starting parameter set {}/{}".format(index + 1, len(parameter_sets))
        )
        run_results = list()
        benchmark = cls()
        benchmark.setup()
        for i in range(0, config["runs"]):
            logging.info("Running {} iter {}/{}".format(cls, i + 1, config["runs"]))
            benchmark.pre_run()
            try:
                run_results.append(benchmark.run(**parameter_set))
            except Exception as e:
                logging.warning(
                    "Exception while runnning benchmark '{}' iter {}: {}".format(
                        cls, iter, str(e)
                    )
                )
            benchmark.post_run()

        benchmark.teardown()
        flat_result = dict()
        for result in run_results:
            for k, v in result.items():
                if k not in metrics:
                    logging.warning(
                        "{} iter {} returned a metric '{}' not described by metrics classback".format(
                            cls, iter, k
                        )
                    )

                if k not in flat_result:
                    flat_result[k] = list()

                flat_result[k].append(v)

        all_results.append(
            {
                "name": "{}.{}".format(cls.__module__, cls.__name__),
                "version": cls.version,
                "metrics": metrics,
                "metadata": metadata,
                "parameters": parameter_set,
                "data": flat_result,
                "config": config,
            }
        )
    return all_results


def get_generic_metadata():
    return {
        "platform": dict(
            zip(
                ("system", "node", "release", "version", "machine", "processor"),
                platform.uname(),
            )
        ),
        "processor": get_processor(),
        "nproc": os.cpu_count(),
        "cpu_online": get_cpu_online(),
        "cpu_possible": get_cpu_possible(),
        "memory_MiB": get_memory(),
        "os-release": get_os_release(),
    }


def get_cpu_possible():
    with open("/sys/devices/system/cpu/possible") as f:
        return f.readlines()[0].strip()


def get_cpu_online():
    with open("/sys/devices/system/cpu/online") as f:
        return f.readlines()[0].strip()


def get_processor():
    with open("/proc/cpuinfo", "r") as f:
        for line in f.readlines():
            if line.startswith("model name"):
                return line.split(":")[1].strip()


def get_memory():
    with open("/proc/meminfo") as f:
        for line in f.readlines():
            if line.startswith("MemTotal:"):
                return int(line.split(":")[1].strip().split(" ")[0]) / 1024.0


def get_os_release():
    data = dict()
    with open("/etc/os-release") as f:
        for line in f.readlines():
            k, v = line.strip().split("=")
            data[k] = v.strip('"')

    return data


def _get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-q",
        "--quiet",
        dest="log_level",
        action="store_const",
        const=logging.ERROR,
        help="Reduce verbosity to errors only",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        dest="log_level",
        action="store_const",
        const=logging.DEBUG,
        help="Increase verbosity",
    )
    parser.add_argument(
        "-c",
        "--config",
        type=pathlib.Path,
        help="The path to YAML or JSON file configuration file",
    )
    parser.add_argument(
        "--benchmarks",
        type=pathlib.Path,
        help="The path to a YAML or JSON file describing which benchmarks should be executed",
    )
    parser.add_argument(
        "--search-path",
        type=pathlib.Path,
        action="append",
        help="The paths to search for benchmarks. May be specified multiple times",
    )
    parser.add_argument(
        "-o", "--output", type=pathlib.Path, help="The output path for the results"
    )

    parser.add_argument(
        "-f",
        "--filter",
        action="append",
        help="Supplemental benchmark filter(s) to apply",
    )

    parser.set_defaults(
        log_level=logging.INFO, search_path=[pathlib.Path("benchmarks").absolute()]
    )
    return parser


def _load_file(path):
    try:
        with open(path, "r") as f:
            return json.load(f)
    except json.JSONDecodeError:
        with open(path, "r") as f:
            return yaml.safe_load(f.read())


def main():
    logging.basicConfig()
    parser = _get_parser()
    args = parser.parse_args()
    logging.getLogger().setLevel(args.log_level)

    config = dict()
    suite = dict()
    if args.config:
        config = _load_file(args.config)

    if args.benchmarks:
        suite = _load_file(args.benchmarks)

    config["search_paths"] = args.search_path
    results = run_benchmarks(config, suite, benchmark_filters=args.filter)
    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f)
    else:
        print(json.dumps(results))


if __name__ == "__main__":
    main()
