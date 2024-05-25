#!/bin/bash
datasets=("mix")
for data in ${datasets[@]}
do
    echo "Train denoise model for bert-base-uncased for $data"
    python train_denoise.py \
        --denoise_data_dir ../data \
        --train_sync True --base_precision float32 \
        --denoise_model_dir ../model \
        --denoise_data=$data --att_pool True --comb att_w_output_v3 --task glue_mrpc  --device cuda:0 \
        --mask_init True --mask_attn True --denoise_model denoiseModelv3 --noise_per_sample 2 \
        --base_model bert-base-uncased --train_eta 100 --test_eta 100 --denoise_size 20000 --num_heads 8 --dim_head 256 \
        --num_layers 6 --denoise_epochs 1 --base_batch_size 16 --denoise_batch_size 12 --d_ff 1024 --clip norm

    etas=(50 100 150)

    for eta in ${etas[@]}
    do
        echo "Test denoise model for eta $eta"
        python test_denoise.py \
            --denoise_data_dir ../data \
            --train_sync True --base_precision float32 \
            --denoise_model_dir ../model \
            --denoise_data=$data --att_pool True --comb att_w_output_v3 --task glue_mrpc --device cuda:0 \
            --mask_init True --mask_attn True --denoise_model denoiseModelv3 --noise_per_sample 2 \
            --base_model bert-base-uncased --train_eta 100 --test_eta=$eta --denoise_size 20000 --num_heads 8 --dim_head 256 \
            --num_layers 6 --denoise_epochs 1 --base_batch_size 16 --denoise_batch_size 12 --d_ff 1024 --clip norm \
            --downstream_task_train_size 10000
        echo "Finishing test denoise model for eta $eta for dataset $data"
    done
done