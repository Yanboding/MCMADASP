#!/bin/bash
#SBATCH -t 48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32
#SBATCH --mem=128G
#  You have to replace Your_account_name below with the name of your account:
#SBATCH -A def-sarhangi

cd /home/dingyanb/projects/def-sarhangi/dingyanb/MCMADASP || exit 1
module load StdEnv/2023 || exit 1
module load meta-farm/1.0.3 || exit 1

# Don't change this line:
autojob.run
