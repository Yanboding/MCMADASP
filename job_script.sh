#!/bin/bash
# Here you should provide the sbatch arguments to be used in all jobs in this serial farm
# It has to contain the runtime switch (either -t or --time):
#SBATCH -t 48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#  You have to replace Your_account_name below with the name of your account:
#SBATCH -A def-sarhangi

cd /home/dingyanb/projects/def-sarhangi/dingyanb/MCMADASP

module load StdEnv/2023
module load scipy-stack
module load gurobi/12.0.0
echo "Threads ${SLURM_CPUS_PER_TASK:-1}" > gurobi.env
source ~/env_gurobi/bin/activate
python -c "import tqdm" >/dev/null 2>&1 || pip install tqdm
# Don't change this line:
task.run
