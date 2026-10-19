"""Compatibility command-line entrypoint for repeated Figure 1 PO training."""
import argparse
import json
from pathlib import Path

from .repeated_po_training import atomic_result, run, validate_params


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--params')
    group.add_argument('--params_file')
    parser.add_argument('--job_id', default='local')
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    params = json.loads(args.params) if args.params is not None else json.loads(Path(args.params_file).read_text())
    run(params, args.job_id)


if __name__ == '__main__':
    main()
