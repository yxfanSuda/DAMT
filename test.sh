#!/usr/bin/env bash
train_file="../Molweni/train.json"
eval_file="../Molweni/dev.json"
test_file="../Molweni/test.json"
dataset_dir="./dataset"
model_dir="./model_dir"
GPU=3

CUDA_VISIBLE_DEVICES=${GPU} python main.py --train_file=$train_file --test_file=$test_file \
                                               --dataset_dir=$dataset_dir \
                                               --eval_pool_size 1 \
                                               --model_path "${model_dir}/model.pt" \
                                               --num_layers 2