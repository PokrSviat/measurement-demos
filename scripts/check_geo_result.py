"""Compare selected fields of a fresh geo case result to the published file."""

import argparse
import json
import math
import sys

CLOSE = (
    ("cuped.incremental", ("cuped", "incremental")),
    ("randomization_interval.low", ("randomization_interval", "low")),
    ("randomization_interval.high", ("randomization_interval", "high")),
    ("additive_randomization_interval.low", ("additive_randomization_interval", "low")),
    ("additive_randomization_interval.high", ("additive_randomization_interval", "high")),
    ("test_cost", ("test_cost",)),
    ("decision.cost_per_incremental", ("decision", "cost_per_incremental")),
)

EXACT = (
    ("decision_interval_shape", ("decision_interval_shape",)),
    ("decision.recommendation", ("decision", "recommendation")),
    ("analysis_plan_sha256", ("analysis_plan_sha256",)),
)


def load(path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def dig(payload, keys):
    value = payload
    for key in keys:
        value = value[key]
    return value


def main():
    parser = argparse.ArgumentParser(
        description="Compare a fresh geo case result to the published one."
    )
    parser.add_argument("new")
    parser.add_argument("published")
    args = parser.parse_args()
    fresh = load(args.new)
    published = load(args.published)
    failed = False
    rows = [(field, keys, True) for field, keys in CLOSE]
    rows.extend((field, keys, False) for field, keys in EXACT)
    for field, keys, close in rows:
        left = dig(fresh, keys)
        right = dig(published, keys)
        same = math.isclose(left, right, rel_tol=1e-6) if close else left == right
        print(f"{field}\t{left}\t{right}\t{'OK' if same else 'DIFF'}")
        failed = failed or not same
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
