# batch_dir=/home/dyf/data_generate/persona-instruct/data/lima/epoch/com/

CUDA_VISIBLE_DEVICES=0,1 python /home/dyf/data_generate/persona-instruct/generate.py \
    --batch_dir /home/dyf/data_generate/persona-instruct/data/lima/response/ \
    --seed_tasks_path /home/dyf/data_generate/persona-instruct/data/lima/persona2/persona_add_lima_persona2_w_vllm.jsonl \
    --roundi 0 \
    --is_vllm \
    --batch_length 50000