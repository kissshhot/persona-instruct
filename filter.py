from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel
import pdb
import torch
import numpy as np
import os
import json
import vllm
from importlib import import_module
import random
import re
import string
import tqdm
import argparse
# os.environ["CUDA_VISIBLE_DEVICES"] = "3"
tokenizer_embedding = AutoTokenizer.from_pretrained('BAAI/bge-small-en-v1.5')
model_embedding = AutoModel.from_pretrained('BAAI/bge-small-en-v1.5') # , device_map={"": "cuda"}
model_embedding.eval()
# todo: 加入embedding过滤和质量过滤方法

def output_log_jsonl(log_file, all_logs):
    with open(log_file, "w") as f:
        for log in all_logs:
            f.write(json.dumps(log) + "\n")

def embedding_filter(txt, sentence_embedding):
    # Tokenize sentences
    encoded_input = tokenizer_embedding(txt, padding=True, truncation=True, return_tensors='pt')
    # for s2p(short query to long passage) retrieval task, add an instruction to query (not add instruction for passages)
    # encoded_input = tokenizer([instruction + q for q in queries], padding=True, truncation=True, return_tensors='pt')

    # Compute token embeddings
    with torch.no_grad():
        model_output = model_embedding(**encoded_input)
        # Perform pooling. In this case, cls pooling.
        txt_embeddings = model_output[0][:, 0]
    # normalize embeddings
    txt_embeddings = torch.nn.functional.normalize(txt_embeddings, p=2, dim=1)
    score_list =[txt_embeddings[0] @ sentence_embedding[i] for i in range(0, len(sentence_embedding))]
    # sentence_embedding = torch.cat((sentence_embedding, txt_embeddings), dim=0)
    if any(x > 0.85 for x in score_list):
        print('embedding不符')
        return False, sentence_embedding
    else:
        print('embedding符合要求')
        sentence_embedding = torch.cat((sentence_embedding, txt_embeddings), dim=0)
        return True, sentence_embedding

def embedding_filter_main(seed_tasks, batch_length):
    all_logs = []
    test_log = []
    question_embedding = torch.load('/home/dyf/data_generate/persona-instruct/embedding/question_embedding.pt')
    questioner_embedding = torch.load('/home/dyf/data_generate/persona-instruct/embedding/questioner_embedding.pt')
    idx = 0
    for t in seed_tasks:
        try:
            source = t['source']
            all_logs.append(t)
        except:
            continue
        question = t['conversations'][0]
        questioner = t['questioner']
        f1, _ = embedding_filter(question, question_embedding)
        f2, _ = embedding_filter(questioner, questioner_embedding)
        if f1 and f2: # filter_output(documents, question) and filter_output(questioner_doc, questioner) and f1 and f2: # and filter_output(respondent_doc, respondent): # and quality_score_vllm(question, model, sampling_params, chat_formatting_function):
            _, question_embedding = embedding_filter(question, question_embedding)
            _, questioner_embedding  = embedding_filter(questioner, questioner_embedding)
            all_logs.append(t)

            output_log_jsonl(os.path.join('/home/dyf/data_generate/persona-instruct/data/lima/epoch/diff/', f"diff_new_instruct_{batch_length}_person2.jsonl"), all_logs)

        else:
            test_ = {}
            test_['idx'] = idx
            test_['result'] = [f1, f2]
            test_log.append(test_)
            output_log_jsonl(os.path.join("/home/dyf/data_generate/persona-instruct/data/lima/wrong/", f"bool_log.jsonl"), test_log)
            continue    
        idx += 1
    return all_logs