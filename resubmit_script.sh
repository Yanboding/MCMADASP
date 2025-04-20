#!/bin/bash
#SBATCH -t 24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=15
#SBATCH --mem=60G
#  You have to replace Your_account_name below with the name of your account:
#SBATCH -A def-sarhangi

# Don't change anything below this line

autojob.run
