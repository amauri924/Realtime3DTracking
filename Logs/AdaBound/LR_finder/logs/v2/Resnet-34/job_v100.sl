#!/bin/bash

# Slurm submission script, 
# GPU job 
# CRIHAN v 1.00 - Jan 2017 
# support@criann.fr

# Not shared resources


# Job name
#SBATCH -J "Pytorch_NU_416x224_k80"

# Batch output file
#SBATCH --output Pytorch_NU_416x224_k80.o%J

# Batch error file
#SBATCH --error Pytorch_NU_416x224_k80.e%J

# GPUs architecture and number
# ----------------------------
# Partition (submission class)
#SBATCH --partition gpu_v100

# GPUs per compute node
#   gpu:4 (maximum) for gpu_k80 
#   gpu:2 (maximum) for gpu_p100 
#SBATCH --gres gpu:1


# CPUs per tack
# k_80 until 7
# p_100 until 13
#SBATCH --cpus-per-task 7
# ----------------------------



# ------------------------
# Job maximum memory (MB)
#SBATCH --mem 32000
# ------------------------

#SBATCH --mail-type ALL
# User e-mail address
##SBATCH --mail-user firstname.name@domain.ext

# environments
# ---------------------------------
module load python3-DL/3.6.9
module load cuda/10.0
module load compilers/gnu/7.3.0
# ---------------------------------


srun python3 ./train.py
