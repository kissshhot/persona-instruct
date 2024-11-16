import numpy as np
import torch
import json
from transformers import AutoModel, AutoTokenizer
from scipy.stats import entropy
from sklearn.decomposition import PCA
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
model_id = "/data1/dyf/model/Mistral-7B-Instruct-v0.3/"
tokenizer = AutoTokenizer.from_pretrained(model_id)
# /home/dyf/data_generate/persona-instruct/data/lima/merged/diff_merged_instruct_20000_person2_round_0.jsonl
# /home/dyf/data_generate/persona-instruct/data/lima/epoch/com/com_new_instruct_round_0.jsonl
# /home/dyf/data_generate/persona-instruct/data/lima/epoch/com/com_new_instruct_round_1.jsonl
seed_tasks_path = "/home/dyf/data_generate/persona-instruct/data/lima/epoch/com/com_new_instruct_round_1.jsonl"
seed_tasks = [json.loads(l) for l in open(seed_tasks_path, "r")]
persona_doc = []
len_sum = 0
i = 0
for tmp in seed_tasks:
    input_id = tokenizer.encode(tmp['conversations'][0])
    l = len(input_id)
    len_sum += l
    i += 1
    if i == 10000:
        break

len_avg = len_sum / 10000
print(len_avg)



