PYTHON=/home/ahmadikalkhorani.1/anaconda3/envs/deepcasa/bin/python

CUDA_VISIBLE_DEVICES=2,3 $PYTHON -u train_pl_AV.py \
    --exp_name "AV" \
    --noise_dataset WHAM \
    --noise_dataset_path /scratch/vahid/WHAM/wham_noise/ \
    --num_dataset_workers 12 \
    --batch_size 4 \
    --add_noise True \
    --audio_only False \
    --start_from_beginning "True" \
    --ckpt_path "none" \
