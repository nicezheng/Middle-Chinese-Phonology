CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node 2 --use_env run_mlm_no_trainer.py \
    --train_file /home/jiangzheng/data/daizhige/out1w.txt \
    --validation_split_percentage 20 \
    --model_name_or_path bert-base-chinese \
    --config_name bert-base-chinese \
    --tokenizer_name bert-base-chinese \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --learning_rate 2e-4 \
    --weight_decay 0.01 \
    --max_train_steps 20000 \
    --gradient_accumulation_steps 4 \
    --num_warmup_steps 5000 \
    --seed 42 \
    --max_seq_length 512 \
    --preprocessing_num_workers 32 \
    --mlm_probability 0.15 \
    --checkpointing_steps 500 \
    --output_dir ./test-mlm \
    --with_tracking \
    --overwrite_cache



#CUDA_VISIBLE_DEVICES=0,1 python run_mlm.py \
#    --model_name_or_path bert-base-chinese \
#    --config_name bert-base-chinese \
#    --tokenizer_name bert-base-chinese \
#    --train_file /home/jiangzheng/data/daizhige/miniout.txt \
#    --validation_split_percentage 10 \
#    --per_device_train_batch_size 8 \
#    --per_device_eval_batch_size 8 \
#    --max_seq_length 512 \
#    --seed 42 \
#    --preprocessing_num_workers 32 \
#    --do_train \
#    --do_eval \
#    --output_dir ./test-mlm