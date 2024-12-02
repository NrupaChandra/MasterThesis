#!/bin/bash

#SBATCH -J Test              # Job name

#
#SBATCH -e /work/home/ng66sume/MasterThesis/%x.err.%j.txt
#SBATCH -o /work/home/ng66sume/MasterThesis/%x.out.%j.txt
#
# CPU Request:
#SBATCH -n 1                                # 1 MPI process/task
#SBATCH -c 4                                # Number of CPU cores (OpenMP threads) per MPI process
#SBATCH --mem-per-cpu=3800                  # Memory in MB per core
#SBATCH -t 02:00:00                         # Job runtime limit (hh:mm:ss)

# GPU Request:
#SBATCH --gres=gpu:v100:1                   # Request 1 GPU of type NVidia V100
# -------------------------------
# Your commands to start the computation:
module purge
module load gcc cuda tensorflow/2.10.0-gpu  # Load necessary modules
cd /work/home/ng66sume/MasterThesis/

# Set environment variables for thread and GPU management:
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Check assigned GPUs (output appears in the error log):
nvidia-smi 1>&2

# Activate your Python environment:
source ~/miniconda3/bin/activate env     

# Run your Python script:
python Helloworld.py                     
EXITCODE=$?

# Job script exits with the program's exit code:
exit $EXITCODE
