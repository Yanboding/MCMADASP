"""Resource-profile and startup checks; no Slurm or Gurobi execution."""

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]


def configurator():
    path = ROOT / "experiments" / "configure_farm.py"
    assert path.is_file(), "the explicit farm profile configurator is missing"
    spec = importlib.util.spec_from_file_location("configure_farm", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def farm(tmp_path):
    worker = """#!/bin/bash
#SBATCH -t 48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH -A custom-account
cd /some/custom/project
task.run
"""
    (tmp_path / "job_script.sh").write_text(worker)
    (tmp_path / "resubmit_script.sh").write_text(worker.replace("task.run", "autojob.run"))
    (tmp_path / "config.h").write_text("NJOBS_MAX=998\ndt_failed=1\nexport NJOBS_MAX dt_failed\n")
    return tmp_path


@pytest.mark.parametrize("profile,tasks,memory,tokens,jobs", [
    ("po", 32, "128G", 33, 151),
    ("evaluation", 16, "64G", 17, 100),
])
def test_profile_updates_both_scripts_and_farm_limit(farm, profile, tasks, memory, tokens, jobs):
    report = configurator().configure_farm(
        farm, profile, jobs=jobs, other_tokens=0, tokens_per_job=tokens)
    for name in ("job_script.sh", "resubmit_script.sh"):
        content = (farm / name).read_text()
        assert f"#SBATCH --ntasks-per-node={tasks}\n" in content
        assert f"#SBATCH --mem={memory}\n" in content
        assert "cpus-per-task" not in content
        assert "#SBATCH -A custom-account" in content
        assert "#SBATCH -t 48:00:00" in content
        assert "cd /some/custom/project" in content
    assert (farm / "config.h").read_text() == f"NJOBS_MAX={jobs}\ndt_failed=1\nexport NJOBS_MAX dt_failed\n"
    assert report["total_tokens"] == jobs * tokens
    assert report["max_jobs"] == 4999 // tokens


def test_strict_token_limit_counts_main_and_other_environments(farm):
    module = configurator()
    assert module.max_safe_jobs(33, 0) == 151
    assert module.max_safe_jobs(33, 16) == 151
    assert module.max_safe_jobs(33, 17) == 150
    assert module.max_safe_jobs(33, 4999) == 0
    assert module.configure_farm(farm, "po", jobs=151, other_tokens=0)["tokens_per_job"] == 33


@pytest.mark.parametrize("kwargs", [
    {"jobs": 152, "other_tokens": 0},
    {"jobs": 151, "other_tokens": 17},
    {"jobs": 1, "other_tokens": 0, "tokens_per_job": 32},
    {"jobs": 0, "other_tokens": 0},
    {"jobs": 1, "other_tokens": -1},
    {"jobs": 1, "other_tokens": 5000},
])
def test_invalid_po_budget_does_not_modify_any_files(farm, kwargs):
    before = {p.name: p.read_bytes() for p in farm.iterdir()}
    with pytest.raises(ValueError):
        configurator().configure_farm(farm, "po", **kwargs)
    assert {p.name: p.read_bytes() for p in farm.iterdir()} == before


def test_evaluation_requires_a_verified_token_estimate(farm):
    with pytest.raises(ValueError, match="tokens_per_job"):
        configurator().configure_farm(farm, "evaluation", jobs=1, other_tokens=0)


def test_configurator_preserves_stricter_existing_farm_cap(farm):
    (farm / "config.h").write_text("NJOBS_MAX=50\n")
    with pytest.raises(ValueError, match="NJOBS_MAX"):
        configurator().configure_farm(farm, "po", jobs=51, other_tokens=0)


@pytest.mark.parametrize("directive", [
    "#SBATCH --cpus-per-task=16 --exclusive",
    "#SBATCH --ntasks=64",
    "#SBATCH -n 64",
])
def test_conflicting_or_combined_resource_directives_fail_without_changes(farm, directive):
    path = farm / "resubmit_script.sh"
    path.write_text(path.read_text() + directive + "\n")
    before = {p.name: p.read_bytes() for p in farm.iterdir()}
    with pytest.raises(ValueError):
        configurator().configure_farm(farm, "po", jobs=1, other_tokens=0)
    assert {p.name: p.read_bytes() for p in farm.iterdir()} == before


def test_malformed_second_script_is_rejected_before_first_script_changes(farm):
    path = farm / "resubmit_script.sh"
    path.write_text(path.read_text().replace("#SBATCH --mem=64G\n", ""))
    before = {p.name: p.read_bytes() for p in farm.iterdir()}
    with pytest.raises(ValueError, match="--mem"):
        configurator().configure_farm(farm, "po", jobs=1, other_tokens=0)
    assert {p.name: p.read_bytes() for p in farm.iterdir()} == before


def test_cli_reports_budget_and_applies_requested_profile(farm):
    result = subprocess.run([
        sys.executable, str(ROOT / "experiments" / "configure_farm.py"),
        "--project-dir", str(farm), "--profile", "po", "--jobs", "150",
        "--other-tokens", "20",
    ], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["total_tokens"] == 4970


def test_worker_startup_does_not_modify_shared_dependencies_or_gurobi_config():
    content = (ROOT / "job_script.sh").read_text()
    assert "pip install" not in content
    assert "gurobi.env" not in content
    assert "srun" not in content
    assert "mpirun" not in content
    assert content.splitlines()[-1] == "task.run"


def test_checked_in_scripts_use_po_resources():
    for name in ("job_script.sh", "resubmit_script.sh"):
        content = (ROOT / name).read_text()
        assert "#SBATCH --ntasks-per-node=32\n" in content
        assert "#SBATCH --mem=128G\n" in content
        assert "cpus-per-task" not in content


def test_worker_stops_on_environment_failure_without_enabling_errexit(tmp_path):
    # Replace site paths and provide shell stubs; never load modules or run a case.
    script = (ROOT / "job_script.sh").read_text()
    script = script.replace("/home/dingyanb/projects/def-sarhangi/dingyanb/MCMADASP", str(tmp_path))
    script = script.replace("source ~/env_gurobi/bin/activate", ":")
    marker = tmp_path / "launched"
    prelude = f"module() {{ return 1; }}\npython() {{ return 0; }}\ntask.run() {{ touch '{marker}'; }}\n"
    result = subprocess.run(["bash", "-c", prelude + script], capture_output=True, text=True)
    assert result.returncode != 0
    assert not marker.exists()
    assert "set -e" not in script
    assert "set -u" not in script


def test_worker_passes_allocated_cpu_budget_and_one_thread_per_internal_worker(tmp_path):
    script = (ROOT / "job_script.sh").read_text()
    script = script.replace("/home/dingyanb/projects/def-sarhangi/dingyanb/MCMADASP", str(tmp_path))
    script = script.replace("source ~/env_gurobi/bin/activate", ":")
    prelude = """module() { return 0; }
python() { return 0; }
task.run() { printf '%s %s %s %s' "$MCMADASP_CPU_BUDGET" "$OMP_NUM_THREADS" "$OPENBLAS_NUM_THREADS" "$MKL_NUM_THREADS"; }
"""
    environment = dict(os.environ, SLURM_CPUS_ON_NODE="32", SLURM_NTASKS="32")
    result = subprocess.run(["bash", "-c", prelude + script], env=environment, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert result.stdout == "32 1 1 1"
