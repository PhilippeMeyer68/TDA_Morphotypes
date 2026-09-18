"""Command line interface: ``tda-morphotypes <step>``."""

from __future__ import annotations

import argparse
import sys
import time

from . import workflow
from .report import write_summary

STEPS = {
    "diagrams": (workflow.compute_all_diagrams, "persistence diagrams of every scan (cached)"),
    "distances": (workflow.compute_all_distances, "Wasserstein distance matrices (cached)"),
    "anomalies": (workflow.run_anomalies, "section 3: anomaly detection"),
    "gdi": (workflow.run_gdi, "section 4: gender discrimination index"),
    "morphotypes": (workflow.run_morphotypes, "section 5: morphotypes"),
    "figures": (workflow.run_figures, "illustrations of section 2 (Figures 1, 2, 5, 6, 9, 10, 15)"),
}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="tda-morphotypes",
        description="Reproduce 'Human Body Shapes Anomaly Detection and Classification Using "
                    "Persistent Homology' (Algorithms, 2023).")
    parser.add_argument("step", choices=[*STEPS, "report", "all"],
                        help="; ".join(f"{k}: {v[1]}" for k, v in STEPS.items())
                        + "; report: compare with the paper and write results/README.md; all: everything")
    parser.add_argument("--data-dir", default="data/raw",
                        help="folder containing male/ and female/ (default: %(default)s)")
    parser.add_argument("--cache-dir", default="data/cache", help="intermediate results (default: %(default)s)")
    parser.add_argument("--results-dir", default="results", help="tables and figures (default: %(default)s)")
    parser.add_argument("-j", "--jobs", type=int, default=None, help="number of processes (default: all CPUs)")
    args = parser.parse_args(argv)

    ws = workflow.Workspace(args.data_dir, args.cache_dir, args.results_dir, args.jobs)
    steps = list(STEPS) if args.step == "all" else [] if args.step == "report" else [args.step]
    start = time.time()
    for step in steps:
        STEPS[step][0](ws)
    if args.step in ("all", "report", "anomalies", "gdi", "morphotypes"):
        check = write_summary(args.results_dir)
        status = [row["status"] for row in check.rows]
        print(f"{status.count('ok')}/{len(status)} values identical to the paper, "
              f"{status.count('known discrepancy')} known discrepancy, {check.failures} mismatch; "
              f"see {args.results_dir}/README.md")
        if check.failures:
            return 1
    print(f"done in {time.time() - start:.0f}s", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
