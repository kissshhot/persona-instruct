from transformers import AutoModelForCausalLM, AutoTokenizer
import pdb
import torch
from datasketch import MinHash, MinHashLSH
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
from prompts.prompt_template_persona2 import persona_generate, persona_generate_simple, persona_diff_instruct_generate
from prompts.score_template import score_template
os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6,7"
model_id = "/data1/dyf/model/Mistral-7B-Instruct-v0.3/"
tokenizer = AutoTokenizer.from_pretrained(model_id)
# model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")

def output_log_jsonl(log_file, all_logs):
    with open(log_file, "w") as f:
        for log in all_logs:
            f.write(json.dumps(log) + "\n")

def calculate_ucb(selected, totals, p):
    C = 3
    n = totals  # 总的数据量
    
    if selected == 0:  # 避免除以零
        ucb = float('inf') # 如果未展示过，则UCB得分为无穷大
    else:
        ucb = C * np.sqrt(2 * np.log(n) / selected)  # 计算UCB得分
    
    return ucb

def filter_output(documents, new_doc):

    # 创建 MinHashLSH 对象，阈值越小标准越高
    lsh = MinHashLSH(threshold=0.5, num_perm=128)

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
        # documents.append(new_doc)
        # print("当前文档列表：")
        # for doc in documents:
        #     print(doc)
        return True
    else:
        print("新文档与已有文档相似，不加入。")
        return False

def quality_score_vllm(result, model, sampling_params, chat_formatting_function):
    prompt = score_template.format(instruct=result)
    t = 0
    while True:
        if t == 10:
            print("score error")
            return False
        try:
            result = use_vllm([prompt], model, sampling_params, chat_formatting_function)
            if len(result.split('### Score:\n')) >= 2:
                result = result.split('### Score:\n')[1].split('\n')[0]
            elif len(result.split('Score:\n')) >= 2:
                result = result.split('Score:\n')[1].split('\n')[0]
            elif len(result.split('### Score: ')) >= 2:
                result = result.split('### Score: ')[1].split('\n')[0]
            elif len(result.split('Score: ')) >= 2:
                result = result.split('Score: ')[1].split('\n')[0]
            if float(result) >= 8:
                print("quality good")
                return True
            else:
                print("quality bad")
                return False
        except:
            t += 1
            continue


def quality_score(result, model):
    prompt = score_template.format(instruct=result)
    conversation = [{"role": "user", "content": prompt}]
    # tools = [get_current_weather]

    # format and tokenize the tool use prompt 
    inputs = tokenizer.apply_chat_template(
                conversation,
                add_generation_prompt=True,
                # return_dict=True,
                return_tensors="pt",
    )

    inputs = inputs.to('cuda')
    times = 0
    while True:
        times += 1
        outputs = model.generate(inputs, max_new_tokens=5000, do_sample=True, temperature=0.7)
        if len(tokenizer.decode(outputs[0][len(inputs[0]):], skip_special_tokens=True).split('### Score:\n')) >= 2:
            result = tokenizer.decode(outputs[0][len(inputs[0]):], skip_special_tokens=True).split('### Score:\n')[1].split('\n')[0]
            break
        elif len(tokenizer.decode(outputs[0][len(inputs[0]):], skip_special_tokens=True).split('Score:\n')) >= 2:
            result = tokenizer.decode(outputs[0][len(inputs[0]):], skip_special_tokens=True).split('Score:\n')[1].split('\n')[0]
            break
        elif len(tokenizer.decode(outputs[0][len(inputs[0]):], skip_special_tokens=True).split('### Score: ')) >= 2:
            result = tokenizer.decode(outputs[0][len(inputs[0]):], skip_special_tokens=True).split('### Score: ')[1].split('\n')[0]
            break
        elif len(tokenizer.decode(outputs[0][len(inputs[0]):], skip_special_tokens=True).split('Score: ')) >= 2:
            result = tokenizer.decode(outputs[0][len(inputs[0]):], skip_special_tokens=True).split('Score: ')[1].split('\n')[0]
            break
        if times == 10:
            return False

    if float(result) >= 7:
        print("quality good")
        return True
    else:
        print("quality bad")
        return False

def dynamic_import_function(function_path):
    '''
    Dynamically import a function from a path string (e.g., "module.submodule.my_function")
    templates.create_prompt_with_huggingface_tokenizer_template
    '''
    module_path, function_name = function_path.rsplit(".", 1)
    module = import_module(module_path)
    function = getattr(module, function_name)
    return function

def create_prompt_with_huggingface_tokenizer_template(messages, tokenizer, add_bos=False):
    formatted_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    if add_bos:
        formatted_text = tokenizer.bos_token + formatted_text
    return formatted_text

def use_vllm(prompts, model, sampling_params, chat_formatting_function):
    
    # chat_formatting_function = dynamic_import_function("templates.create_prompt_with_huggingface_tokenizer_template")
    # model = vllm.LLM(
    #     model=model_id,
    #     tokenizer=model_id,
    #     tokenizer_mode="auto",
    #     tensor_parallel_size=torch.cuda.device_count(),
    #     tokenizer_revision=None, 
    #     revision=None,
    # )
    
    # sampling_params = vllm.SamplingParams(
    #     temperature=0.7,  # greedy decoding
    #     max_tokens=5000,
    #     # stop=args.additional_stop_sequence,
    #     # --additional_stop_sequence',
    #     # type=str,
    #     # nargs="+",
    #     # default=[],
    # )
    # apply chat formatting
    formatted_prompts = []
    for prompt in prompts:
        messages = [{"role": "user", "content": prompt}]
        formatted_prompt = chat_formatting_function(messages, tokenizer, add_bos=False)
        formatted_prompts.append(formatted_prompt)
    prompts = formatted_prompts
            
    outputs = model.generate(prompts, sampling_params)
    outputs = [it.outputs[0].text for it in outputs]
    return outputs[0]

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--batch_dir",
        type=str,
        # required=True,
        default="/home/dyf/data_generate/persona-instruct/data/lima/epoch/diff/",
        help="The directory where the batch is stored.",
    )
    parser.add_argument(
        "--seed_tasks_path",
        type=str,
        # required=True,
        default="/home/dyf/data_generate/persona-instruct/data/lima/persona2/persona_add_lima_persona2_wo_vllm.jsonl",
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
        default=1000,
        help="ins generated each round",
    )
    parser.add_argument(
        "--roundi",
        type=int,
        default=0,
        help="round",
    )
    parser.add_argument(
        "--use_vllm",
        # required=True,
        action="store_true",
        # help="The path to the human written data.",
    )
    return parser.parse_args()

def UCB_sample_record(seed_tasks, batch_length, roundi, is_vllm, model, sampling_params, chat_formatting_function): #记录次数，并加入UCB评分进行采样

    all_logs=[]
    documents = []
    wrong_log = []
    for tmp in seed_tasks:
        documents.append(tmp['conversations'][0])
    if is_vllm == True:
        # chat_formatting_function = dynamic_import_function("templates.create_prompt_with_huggingface_tokenizer_template")
        # model = vllm.LLM(
        #     model=model_id,
        #     tokenizer=model_id,
        #     tokenizer_mode="auto",
        #     tensor_parallel_size=torch.cuda.device_count(),
        #     tokenizer_revision=None, 
        #     revision=None,
        # )
        
        # sampling_params = vllm.SamplingParams(
        #     temperature=0.7,  # greedy decoding
        #     top_p=0.9,
        #     max_tokens=5000,
        #     # stop=args.additional_stop_sequence,
        #     # --additional_stop_sequence',
        #     # type=str,
        #     # nargs="+",
        #     # default=[],
        # )
        x = 0
        for idx in range(batch_length): #len(seed_tasks)
            # if x == 100:
            #     pdb.set_trace()
            # for tmp in seed_tasks:
            #     p = len(tokenizer.encode(tmp['conversations'][0]))
            #     tmp['token'] = p
            # #这里做一个归一化
            # # 提取关键字的值
            # keyword_values = [d['token'] for d in seed_tasks]

            # # 找出关键字值的最小值和最大值
            # min_value = min(keyword_values)
            # max_value = max(keyword_values)

            # # 应用MinMax归一化
            # for d in seed_tasks:
            #     original_value = d['token']
            #     # 归一化公式：(value - min_value) / (max_value - min_value)
            #     normalized_value = (original_value - min_value) / (max_value - min_value) if (max_value - min_value) != 0 else 0
            #     d['p'] = normalized_value * 100
            # # 应用softmax函数
            # # softmax_values = np.exp(keyword_values_array) / np.sum(np.exp(keyword_values_array), axis=0)
            # # for d, value in zip(seed_tasks, softmax_values):
            # #     d['p'] = value
            # for tmp in seed_tasks:
            #     tmp['ucb'] = calculate_ucb(tmp['select_time'], len(seed_tasks), tmp['p'])

            k = 4  # 你想要的前 k 条数据
            # 取出 ucb 值最大的前 k 条数据
            task = random.sample(seed_tasks, k)
            # task = sorted(seed_tasks, key=lambda x: x['ucb'], reverse=True)[:k] # , reverse=True

            #如果达标了才select_time + 1，那么就会一直重复选这k个
            for temp in task:
                temp['select_time'] = temp['select_time'] + 1

            prompt = persona_diff_instruct_generate.format(questioner1=task[0]['questioner'], questioner2=task[1]['questioner'], questioner3=task[2]['questioner'], questioner4=task[3]['questioner'], respondent1=task[0]['respondent'], respondent2=task[1]['respondent'], respondent3=task[2]['respondent'], respondent4=task[3]['respondent'], question1=task[0]['conversations'][0], question2=task[1]['conversations'][0], question3=task[2]['conversations'][0], question4=task[3]['conversations'][0])
            t = 0
            while True:
                if t == 5:
                    # 这里few-shot的例子是乱码，需要移除
                    # set_task = set(task)
                    seed_tasks = [item for item in seed_tasks if item not in task]
                    break
                result = use_vllm([prompt], model, sampling_params, chat_formatting_function)
                try:
                    question = result.split('[New Question]: ')[1]
                    # if len(question.split('\n')) >= 2:
                    #     question = question.split('\n')[0]
                    questioner = result.split('[New Questioner]: ')[1].split('\n')[0]
                    respondent = result.split('[New Respondent]: ')[1].split('\n')[0]
                    break
                except:
                    t += 1
                    continue
            if t == 10:
                continue
                # if len(result.split('[New Question]: ')[1]) >= 2:
                #     question = result.split('[New Question]: ')[1]
                #     if len(question.split('\n')) >= 2:
                #         question = question.split('\n')[0]
                #     break
            if 'environmental' in questioner:
                wrong_log = wrong_log + task
            #     pdb.set_trace()

            if filter_output(documents, question): # and quality_score_vllm(question, model, sampling_params, chat_formatting_function):
                x = 0
                documents.append(question)
                print(result)
                t = {}
                t['questioner'] = questioner
                t['respondent'] = respondent
                t['conversations'] = []
                t['conversations'].append(question)
                t['select_time'] = 1
                all_logs.append(t)
                seed_tasks.append(t)
                # output log at each iteration
                if len(seed_tasks) >= 15000:
                    break
                output_log_jsonl(os.path.join('/home/dyf/data_generate/persona-instruct/data/lima/epoch/diff/', f"diff_new_instruct_{batch_length}_person2_round_{roundi}.jsonl"), all_logs)
                # output log merge
                output_log_jsonl(os.path.join("/home/dyf/data_generate/persona-instruct/data/lima/merged/", f"diff_merged_instruct_{batch_length}_person2_round_{roundi}.jsonl"), seed_tasks)
                # output_log_jsonl(os.path.join("/home/dyf/data_generate/persona-instruct/data/lima/wrong/", f"wrong_log_round_{roundi}.jsonl"), wrong_log)
            else:
                x += 1
                continue
    else:
        model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")
        for idx in range(batch_length): #len(seed_tasks)
            for tmp in seed_tasks:
                p = len(tokenizer.encode(tmp['conversations'][0]))
                tmp['token'] = p
            #这里做一个归一化
            # 提取关键字的值
            keyword_values = [d['token'] for d in seed_tasks]

            # 找出关键字值的最小值和最大值
            min_value = min(keyword_values)
            max_value = max(keyword_values)

            # 应用MinMax归一化
            for d in seed_tasks:
                original_value = d['token']
                # 归一化公式：(value - min_value) / (max_value - min_value)
                normalized_value = (original_value - min_value) / (max_value - min_value) if (max_value - min_value) != 0 else 0
                d['p'] = normalized_value * 100
            # 应用softmax函数
            # softmax_values = np.exp(keyword_values_array) / np.sum(np.exp(keyword_values_array), axis=0)
            # for d, value in zip(seed_tasks, softmax_values):
            #     d['p'] = value
            for tmp in seed_tasks:
                tmp['ucb'] = calculate_ucb(tmp['select_time'], len(seed_tasks), tmp['p'])

            k = 4  # 你想要的前 k 条数据
            # 取出 ucb 值最大的前 k 条数据
            task = sorted(seed_tasks, key=lambda x: x['ucb'], reverse=True)[:k] # , reverse=True

            #如果达标了才select_time + 1，那么就会一直重复选这k个
            for temp in task:
                temp['select_time'] = temp['select_time'] + 1
            
            # continue

            prompt = persona_diff_instruct_generate.format(questioner1=task[0]['questioner'], questioner2=task[1]['questioner'], questioner3=task[2]['questioner'], questioner4=task[3]['questioner'], respondent1=task[0]['respondent'], respondent2=task[1]['respondent'], respondent3=task[2]['respondent'], respondent4=task[3]['respondent'], question1=task[0]['conversations'][0], question2=task[1]['conversations'][0], question3=task[2]['conversations'][0], question4=task[3]['conversations'][0])
            conversation = [{"role": "user", "content": prompt}]
            inputs = tokenizer.apply_chat_template(
                        conversation,
                        add_generation_prompt=True,
                        # return_dict=True,
                        return_tensors="pt",
            )

            inputs = inputs.to('cuda')
            while True:
                outputs = model.generate(inputs, max_new_tokens=5000, do_sample=True, temperature=0.7, top_p=0.9) #现在貌似是gs，后面可能要改成sample
                result = tokenizer.decode(outputs[0][len(inputs[0]):], skip_special_tokens=True)
                try:
                    question = result.split('[New Question]: ')[1]
                    if len(question.split('\n')) >= 2:
                        question = question.split('\n')[0]
                    questioner = result.split('[New Questioner]: ')[1].split('\n')[0]
                    respondent = result.split('[New Respondent]: ')[1].split('\n')[0]
                    break
                except:
                    continue
                # if len(result.split('[New Question]: ')[1]) >= 2:
                #     question = result.split('[New Question]: ')[1]
                #     if len(question.split('\n')) >= 2:
                #         question = question.split('\n')[0]
                #     break
            if filter_output(documents, question) and quality_score(question, model):
                documents.append(question)
                print(tokenizer.decode(outputs[0][len(inputs[0]):], skip_special_tokens=True))
                t = {}
                t['questioner'] = questioner
                t['respondent'] = respondent
                t['conversations'] = []
                t['conversations'].append(question)
                t['select_time'] = 1
                all_logs.append(t)
                seed_tasks.append(t)
                if len(seed_tasks) >= 15000:
                    break
                # output log at each iteration
                output_log_jsonl(os.path.join('/home/dyf/data_generate/persona-instruct/data/lima/epoch/diff/', f"diff_new_instruct_{batch_length}_person2_round_{roundi}.jsonl"), all_logs)
                # output log merge
                output_log_jsonl(os.path.join("/home/dyf/data_generate/persona-instruct/data/lima/merged/", f"diff_merged_instruct_{batch_length}_person2_round_{roundi}.jsonl"), seed_tasks)
            else:
                continue
    return seed_tasks, documents


def main_diff(roundi, seed_tasks, is_vllm, batch_length, model, sampling_params, chat_formatting_function):
    # args = parse_args()
    # if args.use_clf_seed_tasks_only:
    #     seed_tasks = [t for t in seed_tasks if t["is_classification"]]
    if roundi == 0:
        for t in seed_tasks:
            t['select_time'] = 1
    # print(args)
    # print(args.use_vllm)
    # os.makedirs(args.batch_dir, exist_ok=True)
    return UCB_sample_record(seed_tasks, batch_length, roundi, is_vllm, model, sampling_params, chat_formatting_function)

# if __name__ == "__main__":
#     args = parse_args()
#     seed_tasks = [json.loads(l) for l in open(args.seed_tasks_path, "r")]
#     if args.use_clf_seed_tasks_only:
#         seed_tasks = [t for t in seed_tasks if t["is_classification"]]
#     if args.roundi == 0:
#         for t in seed_tasks:
#             t['select_time'] = 1
#     os.makedirs(args.batch_dir, exist_ok=True)

#     do_random_sample_ucb = True
#     if do_random_sample_ucb == True:
#         UCB_sample_record(seed_tasks, args.batch_length)