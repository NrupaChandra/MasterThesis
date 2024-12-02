#!/bin/bash

#SBATCH -J Test
#
#SBATCH -e /work/home/ng66sume/MasterThesis/%x.err.%j.txt
#SBATCH -o /work/home/ng66sume/MasterThesis/%x.out.%j.txt
#
#SBATCH -n 1
#SBATCH --mem-per-cpu=3800
#SBATCH -t 1:00:00
#
#SBATCH -c 16
# GPU Request:
#SBATCH --gres=gpu                # Request 1 GPU of any type
# ---------

module purge
module load gcc/13.1.0 python 
cd /work/home/ng66sume/MasterThesis/

# Check assigned GPUs (output appears in the error log):
nvidia-smi 1>&2

# Activate your Python environment:
source ~/miniconda3/bin/activate env  

python Helloworld.py
EXITCODE=$?

#
exit $EXITCODE
