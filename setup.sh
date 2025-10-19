#!/bin/bash
module load StdEnv/2023
module load scipy-stack
module load gurobi/12.0.0
echo "Threads ${SLURM_CPUS_ON_NODE:-1}" > gurobi.env
source ~/env_gurobi/bin/activate
