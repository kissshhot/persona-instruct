batch_dir=/home/dyf/data_generate/persona-instruct/data/

CUDA_VISIBLE_DEVICES=5,6,7 python /home/dyf/data_generate/persona-instruct/persona_generate_demo.py \
    --batch_dir ${batch_dir} \
    --seed_tasks_path /home/dyf/data_generate/persona-instruct/data/seed_tasks.jsonl \