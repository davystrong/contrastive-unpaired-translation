#!/bin/bash
#SBATCH --job-name=train_cut
#SBATCH --output="/users/40390351/contrastive-unpaired-translation/train_cut_output.txt"
#SBATCH --mail-user=darmstrong34@qub.ac.uk
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --partition=k2-gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --exclude=gpu114
#SBATCH --mem-per-cpu=4G
#SBATCH --time=10:00:00

module load apps/apptainer/1.1.2
# module load nvidia-cuda
# module load libs/nvidia-cuda/11.7.0/bin

dd=/users/40390351
ee=/mnt/scratch2/users/40390351
# cd $dd/Experiments/23-2-2
cd "$dd/contrastive-unpaired-translation"
ln -fs "$dd/ml_notebook/Experiments/Shared/23-11-23.sif" environment.sif
apptainer exec --nv --bind $dd:$dd --bind $ee:$ee ./environment.sif python3 train.py --dataroot ./datasets/mouse --name mouse_CUT --CUT_mode CUT