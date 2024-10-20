from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from datasketch import MinHash, MinHashLSH
import numpy as np
import os
import json
import random
import re
import string
import tqdm
import argparse
from prompts.prompt_template_persona2 import persona_generate, persona_generate_simple, persona_diff_instruct_generate
from prompts.score_template import score_template
os.environ["CUDA_VISIBLE_DEVICES"] = "5,6,7"
model_id = "/data1/dyf/model/Mistral-7B-Instruct-v0.3/"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")

def output_log_jsonl(log_file, all_logs):
    with open(log_file, "w") as f:
        for log in all_logs:
            f.write(json.dumps(log) + "\n")

def calculate_ucb(selected, totals, p):
    C = 1.0
    n = totals  # 总的数据量
    
    if selected == 0:  # 避免除以零
        ucb = float('inf') # 如果未展示过，则UCB得分为无穷大
    else:
        p_hat = p # 指令的长度
        ucb = p_hat + C * np.sqrt(2 * np.log(n) / selected)  # 计算UCB得分
    
    return ucb

def filter_output(documents, new_doc):

    # 创建 MinHashLSH 对象，阈值越大越难选中
    lsh = MinHashLSH(threshold=0.9, num_perm=128)

    # 存储每个文档的 MinHash
    minhashes = {}

    for i, doc in enumerate(documents):
        m = MinHash()
        for word in doc.split():
            m.update(word.encode('utf8'))
        lsh.insert(f'doc_{i}', m)
        minhashes[f'doc_{i}'] = m

    # 为新文档计算 MinHash
    new_minhash = MinHash()
    for word in new_doc.split():
        new_minhash.update(word.encode('utf8'))

    # 查询相似的文档
    result = lsh.query(new_minhash)

    # 判断相似度
    if len(result) == 0:
        print("新文档满足相似度阈值，可以加入。")
        documents.append(new_doc)
        print("当前文档列表：")
        for doc in documents:
            print(doc)
        return True
    else:
        print("新文档与已有文档相似，不加入。")
        return False

def quality_score(result):
    inputs = score_template.format(instruct=result)
    conversation = [{"role": "user", "content": inputs}]
    # tools = [get_current_weather]

    # format and tokenize the tool use prompt 
    inputs = tokenizer.apply_chat_template(
                conversation,
                add_generation_prompt=True,
                # return_dict=True,
                return_tensors="pt",
    )

    inputs = inputs.to('cuda')
    outputs = model.generate(inputs, max_new_tokens=5000)
    result = tokenizer.decode(outputs[0][len(inputs[0]):], skip_special_tokens=True).split('## Score:\n')[1]
    if float(result) >= 6:
        return True
    else:
        return False

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--batch_dir",
        type=str,
        # required=True,
        default="/home/dyf/data_generate/persona-instruct/data/",
        help="The directory where the batch is stored.",
    )
    parser.add_argument(
        "--seed_tasks_path",
        type=str,
        # required=True,
        default="/home/dyf/data_generate/persona-instruct/data/persona_add.jsonl",
        help="The path to the human written data.",
    )
    parser.add_argument(
        "--use_clf_seed_tasks_only",
        action="store_true",
        help="If specified, we will only use the classification seed tasks to prompt new instructions. This will lead to more classification instructions.",
    )
    parser.add_argument(
        "--batch_length",
        type=int,
        default=200,
        help="ins generated each round",
    )
    return parser.parse_args()

def random_sample_record_UCB(seed_tasks, batch_length): #记录次数，并加入UCB评分进行采样

    all_logs=[]
    documents = []
    # for tmp in seed_tasks:
    #     documents.append(tmp['conversations'][0])
    for idx in range(batch_length): #len(seed_tasks)
        for tmp in seed_tasks:
            p = len(tokenizer.encode(tmp['conversations'][0]))
            tmp['ucb'] = calculate_ucb(tmp['select_time'], len(seed_tasks), p)
            documents.append(tmp['conversations'][0])

        k = 4  # 你想要的前 k 条数据
        # 取出 ucb 值最大的前 k 条数据
        task = sorted(seed_tasks, key=lambda x: x['ucb'], reverse=True)[:k]

        #如果达标了才select_time + 1，那么就会一直重复选这k个
        for temp in task:
            temp['select_time'] = temp['select_time'] + 1

        inputs = persona_diff_instruct_generate.format(questioner1=task[0]['questioner'], questioner2=task[1]['questioner'], questioner3=task[2]['questioner'], questioner4=task[3]['questioner'], respondent1=task[0]['respondent'], respondent2=task[1]['respondent'], respondent3=task[2]['respondent'], respondent4=task[3]['respondent'], question1=task[0]['conversations'][0], question2=task[1]['conversations'][0], question3=task[2]['conversations'][0], question4=task[3]['conversations'][0])
        conversation = [{"role": "user", "content": inputs}]
        inputs = tokenizer.apply_chat_template(
                    conversation,
                    add_generation_prompt=True,
                    # return_dict=True,
                    return_tensors="pt",
        )

        inputs = inputs.to('cuda')
        outputs = model.generate(inputs, max_new_tokens=5000, do_sample=True, temperature=0.7, top_p=0.9) #现在貌似是gs，后面可能要改成sample
        result = tokenizer.decode(outputs[0][len(inputs[0]):], skip_special_tokens=True)
        question = result.split('[New Question]: ')[1]
        if len(question.split('\n')) >= 2:
            question = question.split('\n')[0]
        if filter_output(documents, question) and quality_score(question):
            documents.append(question)
            print(tokenizer.decode(outputs[0][len(inputs[0]):], skip_special_tokens=True))
            t = {}
            t['questioner'] = result.split('[New questioner]: ')[1].split('\n')[0]
            t['respondent'] = result.split('[New respondent]: ')[1].split('\n')[0]
            t['conversation'] = []
            t['conversation'].append(question)
            t['select_time'] = 1
            all_logs.append(t)
            seed_tasks.append(t)
            # output log at each iteration
            output_log_jsonl(os.path.join("/home/dyf/data_generate/persona-instruct/data/lima/epoch", f"diff_new_instruct_{batch_length}_person2.jsonl"), all_logs)
            # output log merge
            output_log_jsonl(os.path.join(args.batch_dir, f"diff_new_instruct_{batch_length}_person2.jsonl"), seed_tasks)
        else:
            continue

if __name__ == "__main__":
    args = parse_args()
    seed_tasks = [json.loads(l) for l in open(args.seed_tasks_path, "r")]
    if args.use_clf_seed_tasks_only:
        seed_tasks = [t for t in seed_tasks if t["is_classification"]]
    if args.roundi == 0:
        for t in seed_tasks:
            t['select_time'] = 1
    os.makedirs(args.batch_dir, exist_ok=True)

    do_random_sample_ucb = True
    if do_random_sample_ucb == True:
        new_documents = random_sample_record_UCB(seed_tasks, args.batch_length)