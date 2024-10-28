batch_dir=/home/dyf/data_generate/persona-instruct/data/lima/persona2/

CUDA_VISIBLE_DEVICES=4,5,6,7 python /home/dyf/data_generate/persona-instruct/persona_generate_lima_persona2.py \
    --batch_dir ${batch_dir} \
    --seed_tasks_path /home/dyf/data_generate/persona-instruct/data/lima_train.jsonl \
    --use_vllm true