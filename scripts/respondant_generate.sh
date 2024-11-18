batch_dir=/home/dyf/data_generate/persona-instruct/data/lima/respondant_add/

CUDA_VISIBLE_DEVICES=1,3 python /home/dyf/data_generate/persona-instruct/persona_respondant_generate_lima_persona2.py \
    --batch_dir ${batch_dir} \
    --seed_tasks_path /home/dyf/data_generate/persona-instruct/data/lima_train.jsonl \
    --use_vllm