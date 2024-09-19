#!/bin/bash
sbatch <<EOT
#!/bin/sh
#SBATCH -J $2 #job_name 

#####!/usr/bin/env bash

#SBATCH --output=./logs/job.%j.%x.out
#SBATCH --error=./logs/job.%j.%x.err

# Number of desired cpus (can be in any node):
####SBATCH --ntasks=105

# Number of desired cpus (all in same node):
#SBATCH --cpus-per-task=20

# Amount of RAM needed for this job:
#SBATCH --mem=32gb

# The time the job will be running:
#SBATCH --time=24:00:00

# To use GPUs you have to request them:
#SBATCH --gres=gpu:1

####lshw -C display
####lspci -v | less
nvidia-smi
####export SLURM_ARRAYID
####echo SLURM_ARRAYID: $SLURM_ARRAYID
####echo TASKID: $SLURM_ARRAY_TASK_ID
####sleep 10
module load tensorflow/2.5.0

date
hostname

echo ${@: 3}
echo ${@}
################

# Ensure process affinity is disabled
export SLURM_CPU_BIND=none

# Save the hostname of the allocated nodes
#scontrol show hostnames | tee $(pwd)/hostfile

# Start scoop with python input script
INPUTFILE=$(pwd)/../src/nsga2.py  


hosts=$(srun bash -c hostname)
#time python -m scoop --hostfile ./hostfile -n 105 $INPUTFILE ${@: 3}
#time python -m scoop --hostfile ./hostfile -n 100 ../src/nsga2.py ${@: 3}
time python ../src/nsga2.py ${@: 3}
echo `date` terminado
