#!/bin/bash -x

python modeling/model/main.py \
    --data_dir ./data/splits/ \
    --model_name_or_path google-bert/bert-base-uncased \
    --cache_dir '' \
    --num_train_epochs 5 \
    --token_cls \
    --do_lower_case \
    --seed 0 \
    --token_ratio 0 \
    --training_head 0 \
    --output_dir ./outputs/author_belief \
    --learning_rate 3e-05 \
    --do_train \
    --do_eval \
    --overwrite_output_dir \
    --eval_all_checkpoints
