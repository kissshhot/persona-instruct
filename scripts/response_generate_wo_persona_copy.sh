batch_dir=/home/dyf/data_generate/persona-instruct/data/lima/response/
CUDA_VISIBLE_DEVICES=2,3 python /home/dyf/data_generate/persona-instruct/response_generate_copy.py \
    --seed_tasks_path /home/dyf/data_generate/persona-instruct/data/lima/epoch/com/com_new_instruct_10_round_5.jsonl \
    --batch_dir ${batch_dir}