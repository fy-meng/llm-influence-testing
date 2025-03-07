python src/sft_trainer.py \
    --model_name meta-llama/Llama-3.2-1B \
    --dataset_name datasets/whoqa_train.hf \
    --eval_dataset_name datasets/whoqa_eval.hf \
    --output_dir models/whoqa_llama_1B \
    --dataset_text_field question \
    --use_peft

