#!/bin/bash
# Here you should provide the sbatch arguments to be used in all jobs in this serial farm
# It has to contain the runtime switch (either -t or --time):
#SBATCH -t 24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=15
#SBATCH --mem=60G
#  You have to replace Your_account_name below with the name of your account:
#SBATCH -A def-sarhangi

cd /home/dingyanb/projects/def-sarhangi/dingyanb/MCMADASP

module load StdEnv/2023
module load scipy-stack
module load gurobi/12.0.0
echo "Threads ${SLURM_CPUS_ON_NODE:-1}" > gurobi.env
source ~/env_gurobi/bin/activate
# Don't change this line:
task.run
