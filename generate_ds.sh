#!/bin/bash
#SBATCH --job-name=generate_ds
#SBATCH --output="/users/40390351/contrastive-unpaired-translation/output.txt"
#SBATCH --mail-user=darmstrong34@qub.ac.uk
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --partition=k2-hipri
#SBATCH --mem-per-cpu=4G
#SBATCH --time=1:00:00

module load apps/apptainer/1.1.2
# module load nvidia-cuda
# module load libs/nvidia-cuda/11.7.0/bin

dd=/users/40390351
ee=/mnt/scratch2/users/40390351
# cd $dd/Experiments/23-2-2
cd "$dd/contrastive-unpaired-translation"
ln -fs "$dd/ml_notebook/Experiments/Shared/23-11-23.sif" environment.sif
apptainer exec --nv --bind $dd:$dd --bind $ee:$ee ./environment.sif python3 generate_ds.py