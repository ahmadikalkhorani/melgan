#!/bin/bash
#SBATCH --account=PAA0005
#SBATCH --time=48:00:00
#SBATCH --job-name=AV
#SBATCH --nodes=1
#SBATCH --mem=250G
#SBATCH --output=%x.%j.log
#SBATCH --nodes=1 
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64   
### GPU options ###
#SBATCH --gpus-per-node=1

# DEVICES in the trainer should be the same as ntasks-per-node

# export OMP_NUM_THREADS=$SLURM_CPUS_ON_NODE


module load miniconda3 cuda/11.6.2
source activate deepcasa 


nvidia-smi

echo "------     Starting Time:    `date`     ------"

cd /fs/ess/PAA0005/vahid/Projects/melgan/



LOCAL_DATASET_PATH=/fs/scratch/PAA0005/vahid/Datasets


set -xv   # make script verbose from this point forward -- Galen

echo $SLURM_SUBMIT_DIR/$SLURM_JOB_NAME.$SLURM_JOB_ID.log



srun ~/.conda/envs/deepcasa/bin/python -u -u train_pl_AV.py \
    --exp_name "AV" \
    --audio_dataset SingleChannelAVSpeech \
    --noise_dataset WHAM \
    --audio_dataset_path $LOCAL_DATASET_PATH/AVSpeech/ \
    --noise_dataset_path $LOCAL_DATASET_PATH/WHAM/wham_noise/ \
    --num_dataset_workers 64 \
    --duration 3 \
    --batch_size 24 \
    --add_noise True \
    --audio_only False \
    --start_from_beginning "True" \
    --ckpt_path "none" \
