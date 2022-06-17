import os

if __name__ == "__main__":
    os.system('py transformers/examples/pytorch/summarization/run_summarization.py \
        --model_name_or_path facebook/bart-base \
        --do_train \
        --do_eval \
        --learning_rate 5e-07 \
        --lr_scheduler_type linear \
        --train_file data/train.csv \
        --validation_file data/val.csv \
        --test_file data/test.csv \
        --source_prefix "summarize: " \
        --output_dir data/tst-summarization \
        --overwrite_output_dir \
        --save_steps 10000 \
        --save_total_limit 5 \
        --text_column abstract \
        --summary_column title \
        --per_device_train_batch_size=4 \
        --per_device_eval_batch_size=4 \
        --predict_with_generate')