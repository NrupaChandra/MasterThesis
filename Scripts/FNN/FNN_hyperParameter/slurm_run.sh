#!/bin/bash

#SBATCH -J Test
#
#SBATCH -e /work/home/ng66sume/MasterThesis/Scripts/FNN_V5.1/%x.err.%j.txt
#SBATCH -o /work/home/ng66sume/MasterThesis/Scripts/FNN_V5.1/%x.out.%j.txt
#
#SBATCH -n 1
#SBATCH --mem-per-cpu=3800
#SBATCH -t 50:00:00
#
#SBATCH -c 16
# GPU Request:
#SBATCH --gres=gpu
# ---------

module purge
module load gcc cuda 
cd /work/home/ng66sume/MasterThesis/Scripts/FNN_V5.1/

# Check assigned GPUs (output appears in the error log):
nvidia-smi 1>&2

# Activate your Python environment:
source ~/miniconda3/bin/activate ML

python -u plot.py

EXITCODE=$?

#
exit $EXITCODE
