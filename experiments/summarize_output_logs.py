"""Summarize slurm .out logs: zero-penalty and final-iteration subproblem objective CIs.

For each ``slurm-*.out`` in the output directory, extract
- the ``Zero-penalty subproblem objective mean X +/- Y`` line, and
- the last ``Iteration N, subproblem objective mean X +/- Y`` line,
then write one CSV row per file.

Usage:
    python -m experiments.summarize_output_logs [--output-dir DIR] [--csv PATH]
"""
import argparse
import csv
import glob
import os
import re

DEFAULT_OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "results", "OUTPUT")

ZERO_PENALTY_RE = re.compile(
    r"Zero-penalty subproblem objective mean\s+(-?[\d.]+)\s+\+/-\s+(-?[\d.]+)"
)
ITERATION_RE = re.compile(
    r"Iteration (\d+), subproblem objective mean\s+(-?[\d.]+)\s+\+/-\s+(-?[\d.]+)"
)
GAP_RE = re.compile(r"Gap: (-?[\d.e+-]+)")
CRASH_RE = re.compile(r"Master model optimal solution not found")


def parse_file(path):
    zero_mean = zero_half = None
    final_iter = final_mean = final_half = final_gap = None
    last_iter_line = last_crash_line = -1
    with open(path) as f:
        for line_no, line in enumerate(f):
            match = ZERO_PENALTY_RE.search(line)
            if match:
                zero_mean, zero_half = float(match.group(1)), float(match.group(2))
                continue
            match = ITERATION_RE.search(line)
            if match:
                final_iter = int(match.group(1))
                final_mean, final_half = float(match.group(2)), float(match.group(3))
                last_iter_line = line_no
                continue
            match = GAP_RE.search(line)
            if match:
                final_gap = float(match.group(1))
                continue
            if CRASH_RE.search(line):
                last_crash_line = line_no
    # a crash after the last iteration line means the final case died mid-solve,
    # so the recorded final values are pre-crash garbage
    crashed = last_crash_line > last_iter_line
    return zero_mean, zero_half, final_iter, final_mean, final_half, final_gap, crashed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--csv", default=None,
                        help="destination CSV (default: <output-dir>/subproblem_objective_summary.csv)")
    parser.add_argument("--max-gap", type=float, default=None,
                        help="keep only runs whose final gap is <= this value "
                             "(also drops runs with missing data or negative final mean)")
    args = parser.parse_args()

    csv_path = args.csv or os.path.join(args.output_dir, "subproblem_objective_summary.csv")
    paths = sorted(glob.glob(os.path.join(args.output_dir, "slurm-*.out")))
    if not paths:
        raise SystemExit(f"no slurm-*.out files in {args.output_dir}")

    rows, dropped = [], []
    for path in paths:
        name = os.path.basename(path)
        (zero_mean, zero_half, final_iter, final_mean, final_half,
         final_gap, crashed) = parse_file(path)
        row = [name, zero_mean, zero_half, final_iter, final_mean, final_half, final_gap]
        if args.max_gap is not None:
            converged = (not crashed
                         and final_gap is not None and final_gap <= args.max_gap
                         and final_mean is not None and final_mean > 0)
            if not converged:
                reason = "crashed" if crashed else f"gap={final_gap}"
                dropped.append((name, reason, final_mean))
                continue
        rows.append(row)

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["file", "zero_penalty_mean", "zero_penalty_half_window",
                         "final_iteration", "final_mean", "final_half_window", "final_gap"])
        writer.writerows(rows)

    print(f"saved {csv_path} ({len(rows)} files)")
    if dropped:
        print(f"dropped {len(dropped)} non-converged runs:")
        for name, reason, mean in dropped:
            print(f"  {name}: {reason}, final_mean={mean}")


if __name__ == "__main__":
    main()
