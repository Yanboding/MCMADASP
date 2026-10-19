"""Apply a resource profile and an explicitly budgeted Meta Farm worker limit.

Use only while the target farm is inactive, after preserving its old outputs.
This edits scripts/config.h, never submits jobs or cleans farm state. The caller
must coordinate other queued/running farms for the whole run, not just take a
license-usage snapshot at submission. Evaluation token use must be supplied
after checking its solver-environment lifecycle.
"""

import argparse
import json
from pathlib import Path
import re


PROFILES = {"po": (32, "128G"), "evaluation": (16, "64G")}
TOKEN_LIMIT = 5000


def _integer(value, name, minimum):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


def max_safe_jobs(tokens_per_job, other_tokens):
    """Integer worker bound enforcing total token use strictly below 5000."""
    _integer(tokens_per_job, "tokens_per_job", 1)
    _integer(other_tokens, "other_tokens", 0)
    return max(0, (TOKEN_LIMIT - 1 - other_tokens) // tokens_per_job)


def _replace_directive(content, option, value):
    pattern = rf"^[ \t]*#SBATCH[ \t]+--{option}(?:=|[ \t]+)\S+[^\S\n]*(?:#.*)?$"
    changed, count = re.subn(pattern, f"#SBATCH --{option}={value}", content, flags=re.MULTILINE)
    if count != 1:
        raise ValueError(f"expected exactly one standalone --{option} directive")
    return changed


def configure_farm(project_dir, profile, *, jobs, other_tokens, tokens_per_job=None):
    """Update both scripts and NJOBS_MAX after validating the whole request."""
    if profile not in PROFILES:
        raise ValueError(f"unknown profile: {profile}")
    _integer(jobs, "jobs", 1)
    if tokens_per_job is None:
        if profile != "po":
            raise ValueError("evaluation requires verified tokens_per_job")
        tokens_per_job = 33  # One master plus up to 32 simultaneous subproblem environments.
    token_cap = max_safe_jobs(tokens_per_job, other_tokens)
    if profile == "po" and tokens_per_job < 33:
        raise ValueError("PO tokens_per_job must include the master and 32 workers (>= 33)")
    if jobs > token_cap:
        raise ValueError(f"jobs={jobs} exceeds the strict token budget (maximum {token_cap})")

    directory = Path(project_dir)
    config_path = directory / "config.h"
    config = config_path.read_text()
    cap_pattern = r"^[ \t]*NJOBS_MAX=(\d+)[ \t]*(?:#.*)?$"
    caps = re.findall(cap_pattern, config, flags=re.MULTILINE)
    if len(caps) != 1:
        raise ValueError("expected exactly one literal NJOBS_MAX in config.h")
    if jobs > int(caps[0]):
        raise ValueError(f"jobs={jobs} exceeds existing NJOBS_MAX={caps[0]}")

    tasks, memory = PROFILES[profile]
    updates = {}
    for name in ("job_script.sh", "resubmit_script.sh"):
        path = directory / name
        content = path.read_text()
        content = _replace_directive(content, "ntasks-per-node", tasks)
        content = _replace_directive(content, "mem", memory)
        content = re.sub(
            r"^[ \t]*#SBATCH[ \t]+(?:--cpus-per-task(?:=|[ \t]+)|-c[ \t]*)\S+[^\S\n]*(?:#.*)?$\n?",
            "", content, flags=re.MULTILINE)
        if re.search(
            r"^[ \t]*#SBATCH[^\n]*(?:--ntasks(?:=|[ \t]|$)|(?<!\S)-n(?:[ \t]|\d)|"
            r"--cpus-per-task|(?<!\S)-c(?:[ \t]|\d))", content, flags=re.MULTILINE
        ):
            raise ValueError(f"{name} has a conflicting or combined CPU/task directive")
        updates[path] = content
    updates[config_path] = re.sub(cap_pattern, f"NJOBS_MAX={jobs}", config, flags=re.MULTILINE)
    for path, content in updates.items():
        path.write_text(content)
    return {
        "profile": profile,
        "ntasks_per_node": tasks,
        "memory": memory,
        "jobs": jobs,
        "tokens_per_job": tokens_per_job,
        "other_tokens": other_tokens,
        "total_tokens": jobs * tokens_per_job + other_tokens,
        "max_jobs": min(token_cap, int(caps[0])),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-dir", type=Path, required=True)
    parser.add_argument("--profile", choices=PROFILES, required=True)
    parser.add_argument("--jobs", type=int, required=True, help="Farm workers; also sets NJOBS_MAX")
    parser.add_argument("--other-tokens", type=int, required=True,
                        help="Reserved token use of other concurrent/queued jobs")
    parser.add_argument("--tokens-per-job", type=int,
                        help="Verified environment upper bound; PO defaults to 33, evaluation requires it")
    args = parser.parse_args(argv)
    try:
        report = configure_farm(**vars(args))
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    print(json.dumps(report, indent=2, sort_keys=True))
    return report


if __name__ == "__main__":
    main()
