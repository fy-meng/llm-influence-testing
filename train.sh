#!/bin/bash

#SBATCH --job-name=train
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1

python src/sft_trainer.py \
    --model_name meta-llama/Llama-3.2-1B \
    --dataset_name datasets/whoqa_train.hf \
    --eval_dataset_name datasets/whoqa_eval.hf \
    --output_dir models/whoqa_llama_1B \
    --dataset_text_field prompt \
    --use_peft

