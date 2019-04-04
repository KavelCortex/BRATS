#!/bin/bash
#
#SBATCH --job-name=jwang96_BRATS
#SBATCH --output=res.out
#SBATCH --error=res.err
#
# Number of tasks needed for this job. Generally, used with MPI jobs
#SBATCH --ntasks=1
#SBATCH --partition=pascalnodes
#
# Time format = HH:MM:SS, DD-HH:MM:SS
#SBATCH --time=48:00:00
#
# Number of CPUs allocated to each task. 
#SBATCH --cpus-per-task=4
#
# Mimimum memory required per allocated  CPU  in  MegaBytes. 
#SBATCH --mem-per-cpu=8192
#
# Send mail to the email address when the job fails
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=jwang96@uab.edu

#Set your environment here
module load Anaconda3/5.3.0
module load cuDNN/6.0-CUDA-8.0.61
module load Tensorflow/1.2.0-intel-2017a-Python-3.6.1
conda install keras


#Run your commands here
srun --gres=gpu:4 python main.py