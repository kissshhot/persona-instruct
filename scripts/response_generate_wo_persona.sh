# batch_dir=/home/dyf/data_generate/persona-instruct/data/lima/persona2/
CUDA_VISIBLE_DEVICES=4,5,6,7 python /home/dyf/data_generate/persona-instruct/response_generate.py \
    --seed_tasks_path /home/dyf/data_generate/persona-instruct/data/lima/wo_persona/com_new_instruct_round_1.jsonl \