PYTHON=/home/ahmadikalkhorani.1/anaconda3/envs/deepcasa/bin/python

CUDA_VISIBLE_DEVICES=1 $PYTHON -u train_pl.py \
    --exp_name "AO-STFT" \
    --num_dataset_workers 32 \
    --n_mel_channels 513 \
    --mel False \
    --batch_size 8 \
    --audio_only True \
    --start_from_beginning "True" \
    --ckpt_path "none"
