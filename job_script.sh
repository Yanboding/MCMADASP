#!/bin/bash
# Here you should provide the sbatch arguments to be used in all jobs in this serial farm
# It has to contain the runtime switch (either -t or --time):
#SBATCH -t 48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32
#SBATCH --mem=128G
#  You have to replace Your_account_name below with the name of your account:
#SBATCH -A def-sarhangi

cd /home/dingyanb/projects/def-sarhangi/dingyanb/MCMADASP || exit 1

module load StdEnv/2023 || exit 1
module load scipy-stack/2026a || exit 1
module load gurobi/12.0.0 || exit 1
module load meta-farm/1.0.3 || exit 1
source ~/env_gurobi/bin/activate || exit 1
python -c "import gurobipy, numpy, scipy, tqdm" || exit 1

# One Python case at a time; its internal workers use the allocation's CPUs.
export MCMADASP_CPU_BUDGET="${SLURM_CPUS_ON_NODE:-${SLURM_NTASKS:-1}}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
# Do not enable shell errexit: Meta Farm sources single_case.sh to record failures.
# Don't change this line:
task.run
