#!/usr/bin/env bash
train_file="../Molweni/train.json"
eval_file="../Molweni/dev.json"
test_file="../Molweni/test.json"
dataset_dir="./dataset"
model_dir="./model_dir"

if [ ! -d "${model_dir}" ]; then mkdir -p "${model_dir}"; fi
GPU=3
CUDA_VISIBLE_DEVICES=${GPU}   nohup python -u  main.py --train_file=$train_file --eval_file=$eval_file --test_file=$test_file \
                                    --dataset_dir=$dataset_dir  \
                                    --epoches 20 --batch_size 150 --pool_size 150 \
                                    --eval_pool_size 1 --report_step 30 \
                                    --save_model  --do_train \
                                    --model_path "${model_dir}/model.pt" \
                                    --num_layers 2 \
                                    --num_heads 4 \
                                    --dropout 0.5 \
                                    --seed 65534 >DAMT.log 2>&1 &