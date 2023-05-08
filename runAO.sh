PYTHON=/home/ahmadikalkhorani.1/anaconda3/envs/deepcasa/bin/python

CUDA_VISIBLE_DEVICES=0 $PYTHON -u train_pl.py \
    --exp_name "AO" \
    --num_dataset_workers 32 \
    --batch_size 16 \
    --audio_only True \
    --start_from_beginning "False" \
    --ckpt_path "/scratch/vahid/melgan/checkpoints/melgan-last.ckpt"
