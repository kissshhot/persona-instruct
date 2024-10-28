batch_dir=/home/dyf/data_generate/persona-instruct/data/lima/epoch/diff/

CUDA_VISIBLE_DEVICES=5,6,7 python /home/dyf/data_generate/persona-instruct/persona_diff_instruct_generate_demo_lima_persona2.py \
    --batch_dir ${batch_dir} \
    --seed_tasks_path /home/dyf/data_generate/persona-instruct/data/lima/persona2/persona_add_lima_persona2_wo_vllm.jsonl \
    --roundi 0