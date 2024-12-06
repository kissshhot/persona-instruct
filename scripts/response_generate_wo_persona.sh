batch_dir=/home/dyf/data_generate/persona-instruct/data/lima/response/
CUDA_VISIBLE_DEVICES=0,1 python /home/dyf/data_generate/persona-instruct/response_generate.py \
    --seed_tasks_path /home/dyf/data_generate/persona-instruct/data/lima/epoch/com/com_new_instruct_10_round_5.jsonl \
    --batch_dir ${batch_dir}