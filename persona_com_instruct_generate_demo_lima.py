from transformers import AutoModelForCausalLM, AutoTokenizer
from datasketch import MinHash, MinHashLSH
import numpy as np
import torch
import os
import json
import random
import re
import string
import tqdm
import argparse
from prompts.prompt_template import persona_generate, persona_generate_simple, persona_com_instruct_generate
from prompts.score_template import score_template
os.environ["CUDA_VISIBLE_DEVICES"] = "5,6,7"
model_id = "/data1/dyf/model/Mistral-7B-Instruct-v0.3/"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")
# def get_current_weather(location: str, format: str):
#     """
#     Get the current weather

#     Args:
#         location: The city and state, e.g. San Francisco, CA
#         format: The temperature unit to use. Infer this from the users location. (choices: ["celsius", "fahrenheit"])
#     """
#     pass
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
        default="/home/dyf/data_generate/persona-instruct/data/lima/",
        help="The directory where the batch is stored.",
    )
    parser.add_argument(
        "--seed_tasks_path",
        type=str,
        # required=True,
        default="/home/dyf/data_generate/persona-instruct/data/lima/persona_add_lima.jsonl",
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
    parser.add_argument(
        "--roundi",
        type=int,
        default=0,
        help="round",
    )
    parser.add_argument(
        "--th",
        type=float,
        default=5.0,
        help="th of ucb",
    )
    return parser.parse_args()

def random_sample(seed_tasks, batch_length, roundi): #随机选择数据进行generate

    all_logs=[]
    documents = []
    for tmp in seed_tasks:
        documents.append(tmp['conversations'][0])
    task = random.sample(seed_tasks, batch_length)
    for idx in range(len(task)): #len(seed_tasks)
        # dialogue = ''
        # for tx in seed_tasks[idx]['conversations']:
        #     dialogue += tx + '\n'
        description = task[idx]['persona']
        question = task[idx]['conversations'][0]
        # dialogue = seed_tasks[idx]['instruction'] # + '\n' + seed_tasks[idx]['instances'][0]['input'] + '\n' + seed_tasks[idx]['instances'][0]['output']
        inputs = persona_com_instruct_generate.format(description=description, question=question)
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
        outputs = model.generate(inputs, max_new_tokens=5000, do_sample=True, temperature=0.7, top_p=0.9) #现在貌似是gs，后面可能要改成sample
        result = tokenizer.decode(outputs[0][len(inputs[0]):], skip_special_tokens=True)
        question = result.split('[New Question]: ')[1]
        if filter_output(documents, question) and quality_score(question):
            documents.append(question)
            print(tokenizer.decode(outputs[0][len(inputs[0]):], skip_special_tokens=True))
            t = {}
            t['persona'] = result.split('[New Description]: ')[1].split('\n')[0]
            t['conversation'] = []
            t['conversation'].append(question)
            # t['id'] = 'com_1'
            # t['source'] = 'sharegpt'
            # t['conversations'] = []
            # t['conversations'].append({"from":"user", "value": })
            all_logs.append(t)
            output_log_jsonl(os.path.join("/home/dyf/data_generate/persona-instruct/data/lima/epoch", f"com_new_instruct_{batch_length}_round_{roundi}.jsonl"), all_logs) 
            # output log at each iteration
            all_logs = all_logs + seed_tasks
            output_log_jsonl(os.path.join(args.batch_dir, f"com_new_instruct_{batch_length}_round_{roundi}.jsonl"), all_logs)
        else:
            continue

def random_sample_record(seed_tasks, batch_length, roundi): #选择过的数据会被记录选择次数

    all_logs=[]
    documents = []
    for tmp in seed_tasks:
        documents.append(tmp['conversations'][0])
    unselected_tasks = [tmp for tmp in seed_tasks if tmp['select_time'] <= 3]
    # 验证一下原始数据的select_time有没有变
    task = random.sample(unselected_tasks, batch_length)
    for temp in task:
        temp['selecte_time'] = temp['selecte_time'] + 1
    for idx in range(len(task)): #len(seed_tasks)
        # dialogue = ''
        # for tx in seed_tasks[idx]['conversations']:
        #     dialogue += tx + '\n'
        description = task[idx]['persona']
        question = task[idx]['conversations'][0]
        # dialogue = seed_tasks[idx]['instruction'] # + '\n' + seed_tasks[idx]['instances'][0]['input'] + '\n' + seed_tasks[idx]['instances'][0]['output']
        inputs = persona_com_instruct_generate.format(description=description, question=question)
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
        outputs = model.generate(inputs, max_new_tokens=5000, do_sample=True, temperature=0.7, top_p=0.9) #现在貌似是gs，后面可能要改成sample
        result = tokenizer.decode(outputs[0][len(inputs[0]):], skip_special_tokens=True)
        question = result.split('[New Question]: ')[1]
        if filter_output(documents, question) and quality_score(question):
            documents.append(question)
            print(tokenizer.decode(outputs[0][len(inputs[0]):], skip_special_tokens=True))
            t = {}
            t['persona'] = result.split('[New Description]: ')[1].split('\n')[0]
            t['conversation'] = []
            t['conversation'].append(question)
            t['select_time'] = 1
            # t['id'] = 'com_1'
            # t['source'] = 'sharegpt'
            # t['conversations'] = []
            # t['conversations'].append({"from":"user", "value": })
            all_logs.append(t)
            # output log at each iteration
            output_log_jsonl(os.path.join("/home/dyf/data_generate/persona-instruct/data/lima/epoch", f"com_new_instruct_{batch_length}_round_{roundi}.jsonl"), all_logs) 

            all_logs = all_logs + seed_tasks
            output_log_jsonl(os.path.join(args.batch_dir, f"com_new_instruct_{batch_length}_round_{roundi}.jsonl"), all_logs)
        else:
            continue

def random_sample_record_UCB(seed_tasks, batch_length, roundi, th): #记录次数，并加入UCB评分进行采样

    all_logs=[]
    documents = []
    # for tmp in seed_tasks:
    #     documents.append(tmp['conversations'][0])
    for tmp in seed_tasks:
        p = len(tokenizer.encode(tmp['conversations'][0]))
        tmp['ucb'] = calculate_ucb(tmp['select_time'], len(seed_tasks), p)
        documents.append(tmp['conversations'][0])

    # k = 2  # 你想要的前 k 条数据
    # 取出 ucb 值最大的前 k 条数据
    # task = sorted(seed_tasks, key=lambda x: x['ucb'], reverse=True)[:k]
    unselected_tasks = [temp for temp in seed_tasks if temp['ucb'] >= th]
    if len(unselected_tasks) < batch_length:
        raise ValueError("len(unselected_tasks) < batch_length")
    task = random.sample(unselected_tasks, batch_length)
    for temp in task:
        temp['selecte_time'] = temp['selecte_time'] + 1
    for idx in range(len(task)): #len(seed_tasks)
        # dialogue = ''
        # for tx in seed_tasks[idx]['conversations']:
        #     dialogue += tx + '\n'
        description = task[idx]['persona']
        question = task[idx]['conversations'][0]
        # dialogue = seed_tasks[idx]['instruction'] # + '\n' + seed_tasks[idx]['instances'][0]['input'] + '\n' + seed_tasks[idx]['instances'][0]['output']
        inputs = persona_com_instruct_generate.format(description=description, question=question)
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
        outputs = model.generate(inputs, max_new_tokens=5000, do_sample=True, temperature=0.7, top_p=0.9) #现在貌似是gs，后面可能要改成sample
        result = tokenizer.decode(outputs[0][len(inputs[0]):], skip_special_tokens=True)
        question = result.split('[New Question]: ')[1]
        if filter_output(documents, question) and quality_score(question):
            documents.append(question)
            print(tokenizer.decode(outputs[0][len(inputs[0]):], skip_special_tokens=True))
            t = {}
            t['persona'] = result.split('[New Description]: ')[1].split('\n')[0]
            t['conversation'] = []
            t['conversation'].append(question)
            t['select_time'] = 1
            # t['id'] = 'com_1'
            # t['source'] = 'sharegpt'
            # t['conversations'] = []
            # t['conversations'].append({"from":"user", "value": })
            all_logs.append(t)
            # output log at each iteration
            output_log_jsonl(os.path.join("/home/dyf/data_generate/persona-instruct/data/lima/epoch", f"com_new_instruct_{batch_length}_round_{roundi}.jsonl"), all_logs) 

            all_logs = all_logs + seed_tasks
            output_log_jsonl(os.path.join(args.batch_dir, f"com_new_instruct_{batch_length}_round_{roundi}.jsonl"), all_logs)
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

    do_random_sample = True
    if do_random_sample == True:
        new_documents = random_sample_record(seed_tasks, args.batch_length, args.roundi)
    do_random_sample_ucb = True
    if do_random_sample_ucb == True:
        new_documents = random_sample_record_UCB(seed_tasks, args.batch_length, args.roundi, args.th)
    # documents = []
    # for tmp in seed_tasks:
    #     documents.append(tmp['conversations'][0])