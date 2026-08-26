#!/bin/bash
module load StdEnv/2023
module load scipy-stack
module load gurobi/12.0.0
echo "Threads ${SLURM_CPUS_ON_NODE:-1}" > gurobi.env
source ~/env_gurobi/bin/activate

./single_case.sh ./table.dat 1 
python generate_params.py train case_study_099_mixture_geometric_proposal_095 \
    --num-init-states 500 --init-state-seed 12345