#!/usr/bin/env python3
#
# SPDX-License-Identifier: GPL-2.0-only
# SPDX-FileCopyrightText: 2026 Kienan Stewart <kstewart@efficios.com>
#


class BenchmarkBase:

    version = None

    def default_parameter_sets():
        """
        Return a list of dicts with parameters for the run()
        """
        return [dict()]

    def metadata():
        """
        Return a dict with benchmark specific metadata
        """
        return dict()

    def metrics():
        """
        Should return a dict with information about the metrics that run() returns,
        including units, expected interpretation (e.g. lower is better), ...
        """
        return dict()

    def setup(self):
        pass

    def teardown(self):
        pass

    def pre_run(self):
        pass

    def run(self):
        """
        Should return a dict of {metric: value, ...} for a single test run
        """
        return dict()

    def post_run(self):
        pass

    def __del__(self):
        self.post_run()
        self.teardown()
