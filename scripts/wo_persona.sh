# batch_dir=/home/dyf/data_generate/persona-instruct/data/lima/epoch/com/

CUDA_VISIBLE_DEVICES=4,5,6,7 python /home/dyf/data_generate/persona-instruct/baseline_wo_persona.py \
    --seed_tasks_path /home/dyf/data_generate/persona-instruct/data/lima/wo_persona/com_new_instruct_round_0.jsonl \
    --roundi 0 \
    --is_vllm \
    --batch_length 20000