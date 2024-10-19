batch_dir=/home/dyf/data_generate/persona-instruct/data/lima/

CUDA_VISIBLE_DEVICES=3,4,5 python /home/dyf/data_generate/persona-instruct/persona_generate_lima.py \
    --batch_dir ${batch_dir} \
    --seed_tasks_path /home/dyf/data_generate/persona-instruct/data/lima_train.jsonl \