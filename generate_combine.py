import argparse
import json
import os
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel
from persona_com_instruct_generate_demo_lima_persona2 import main_com
from persona_diff_instruct_generate_demo_lima_persona2 import main_diff
import vllm
import pdb
import random
from importlib import import_module
import torch
from prompts.prompt_template_persona2 import persona_generate, persona_generate_simple, persona_diff_instruct_generate, persona_diff_instruct_generate_simple, persona_diff_instruct_generate_re, persona_diff_instruct_generate_wo_question
from prompts.score_template import score_template
model_id = "/data1/dyf/model/Mistral-7B-Instruct-v0.3/"
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer_embedding = AutoTokenizer.from_pretrained('BAAI/bge-small-en-v1.5')
model_embedding = AutoModel.from_pretrained('BAAI/bge-small-en-v1.5') # , device_map={"": "cuda"}
model_embedding.eval()
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
# from persona_diff_instruct_generate_demo_lima_persona2 import main_diff
model_id = "/data1/dyf/model/Mistral-7B-Instruct-v0.3/"

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--batch_dir",
        type=str,
        # required=True,
        default="/home/dyf/data_generate/persona-instruct/data/lima/epoch/com/",
        help="The directory where the batch is stored.",
    )
    parser.add_argument(
        "--seed_tasks_path",
        type=str,
        # required=True,
        default="/home/dyf/data_generate/persona-instruct/data/lima/merged/diff_merged_instruct_12000_person2_round_0.jsonl",
        help="The path to the human written data.",
    )
    parser.add_argument(
        "--use_clf_seed_tasks_only",
        action="store_true",
        help="If specified, we will only use the classification seed tasks to prompt new instructions. This will lead to more classification instructions.",
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
    parser.add_argument(
        "--is_vllm",
        action="store_true",
        # required=True,
        # default=True,
        # help="The path to the human written data.",
    )
    parser.add_argument(
        "--batch_length",
        type=int,
        default=10,
        help="ins generated each round",
    )
    return parser.parse_args()

def output_log_jsonl(log_file, all_logs):
    with open(log_file, "w") as f:
        for log in all_logs:
            f.write(json.dumps(log) + "\n")

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
    if any(x > 0.8 for x in score_list):
        print('embedding不符')
        return False, sentence_embedding
    else:
        print('embedding符合要求')
        sentence_embedding = torch.cat((sentence_embedding, txt_embeddings), dim=0)
        return True, sentence_embedding

def UCB_sample_record(seed_tasks, batch_length, roundi, is_vllm, model, sampling_params, chat_formatting_function): #记录次数，并加入UCB评分进行采样

    all_logs=[]
    raw_logs = []
    documents = []
    questioner_doc = []
    respondent_doc = []
    test_log = []
    wrong_log = []
    # for tmp in seed_tasks:
    #     documents.append(tmp['conversations'][0])
    # for tmp in seed_tasks:
    #     questioner_doc.append(tmp['questioner'])
    # for tmp in seed_tasks:
    #     respondent_doc.append(tmp['respondent'])
    question_embedding = torch.load('/home/dyf/data_generate/persona-instruct/embedding/question_embedding.pt')
    questioner_embedding = torch.load('/home/dyf/data_generate/persona-instruct/embedding/questioner_embedding.pt')
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
        for idx in range(1000000): #len(seed_tasks)
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
            # for temp in task:
            #     temp['select_time'] = temp['select_time'] + 1
            prompt = persona_diff_instruct_generate_wo_question.format(questioner1=task[0]['questioner'], questioner2=task[1]['questioner'], questioner3=task[2]['questioner'], questioner4=task[3]['questioner'], respondent1=task[0]['respondent'], respondent2=task[1]['respondent'], respondent3=task[2]['respondent'], respondent4=task[3]['respondent'], question1=task[0]['conversations'][0], question2=task[1]['conversations'][0], question3=task[2]['conversations'][0], question4=task[3]['conversations'][0])
            # prompt = persona_diff_instruct_generate_simple.format(questioner1=task[0]['questioner'], questioner2=task[1]['questioner'], questioner3=task[2]['questioner'], question1=task[0]['conversations'][0], question2=task[1]['conversations'][0], question3=task[2]['conversations'][0])
            et = 0
            while True:
                # if et == 1:
                #     # 这里few-shot的例子是乱码，需要移除
                #     # set_task = set(task)
                #     # seed_tasks = [item for item in seed_tasks if item not in task]
                #     break
                result = use_vllm([prompt], model, sampling_params, chat_formatting_function)
                try:
                    question = result.split('[New Question]: ')[1].split('[Collaborative Relationship]: ')[0].strip('"')
                    # if len(question.split('\n')) >= 2:
                    #     question = question.split('\n')[0]
                    # questioner = result.split('### questioner:\n')[1].split('\n### respondent:\n')[0].strip()
                    questioner = result.split('[New Questioner]: ')[1].split('[New Question]: ')[0].strip('"')
                    # respondent = result.split('[New Respondent]: ')[1].split('[Collaborative Relationship]: ')[0].strip('"')
                    # Relationship = result.split('[Collaborative Relationship]: ')[1]
                    break
                except:
                    et += 1
                    break
            if et == 1:
                continue
                # if len(result.split('[New Question]: ')[1]) >= 2:
                #     question = result.split('[New Question]: ')[1]
                #     if len(question.split('\n')) >= 2:
                #         question = question.split('\n')[0]
                #     break
            # if 'environmental' in questioner:
            #     wrong_log = wrong_log + task
            #     pdb.set_trace()
            print(prompt)
            print(result)
            pdb.set_trace()
            f1, _ = embedding_filter(question, question_embedding)
            f2, _ = embedding_filter(questioner, questioner_embedding)
            t = {}
            t['questioner'] = questioner
            # t['respondent'] = respondent
            # t['Relationship'] = respondent
            t['conversations'] = []
            t['conversations'].append(question)
            raw_logs.append(t)
            if f1 and f2: # filter_output(documents, question) and filter_output(questioner_doc, questioner) and f1 and f2: # and filter_output(respondent_doc, respondent): # and quality_score_vllm(question, model, sampling_params, chat_formatting_function):
                _, question_embedding = embedding_filter(question, question_embedding)
                _, questioner_embedding  = embedding_filter(questioner, questioner_embedding)
                documents.append(question)
                questioner_doc.append(questioner)
                # respondent_doc.append(respondent)
                print(result)
                # t = {}
                # t['questioner'] = questioner
                # # t['respondent'] = respondent
                # # t['Relationship'] = respondent
                # t['conversations'] = []
                # t['conversations'].append(question)
                # t['select_time'] = 1
                if idx <= 1000:
                    wrong_log = wrong_log + task + [t]
                all_logs.append(t)
                # seed_tasks.append(t)
                # output log at each iteration
                if len(all_logs) >= batch_length:
                    break
                output_log_jsonl(os.path.join('/home/dyf/data_generate/persona-instruct/data/lima/epoch/diff/', f"diff_new_instruct_{batch_length}_person2_round_{roundi}.jsonl"), all_logs)
                # output log merge
                output_log_jsonl(os.path.join("/home/dyf/data_generate/persona-instruct/data/lima/merged/", f"diff_merged_instruct_{batch_length}_person2_round_{roundi}.jsonl"), seed_tasks)
                output_log_jsonl(os.path.join("/home/dyf/data_generate/persona-instruct/data/lima/wrong/", f"check_log_round_{roundi}.jsonl"), wrong_log)
                # output_log_jsonl(os.path.join("/home/dyf/data_generate/persona-instruct/data/lima/wrong/", f"bool_log_round_{roundi}.jsonl"), test_log)
            else:
                test_ = {}
                test_['id'] = idx
                test_['result'] = [f1, f2]
                test_log.append(test_)
                output_log_jsonl(os.path.join("/home/dyf/data_generate/persona-instruct/data/lima/wrong/", f"bool_log_round_{roundi}.jsonl"), test_log)
                continue
        output_log_jsonl(os.path.join("/home/dyf/data_generate/persona-instruct/data/lima/raw_data/", f"diff_raw_instruct_{batch_length}_person2_round_{roundi}.jsonl"), raw_logs)
        output_log_jsonl(os.path.join("/home/dyf/data_generate/persona-instruct/data/lima/merged/", f"diff_merged_instruct_{batch_length}_person2_round_{roundi}.jsonl"), seed_tasks + all_logs)
    all_logs = seed_tasks + all_logs
    return all_logs, documents

if __name__ == "__main__":
    args = parse_args()
    args.is_vllm = True
    all_logs = []
    roundi = args.roundi
    seed_tasks = [json.loads(l) for l in open(args.seed_tasks_path, "r")]
    # task2 = [json.loads(l) for l in open('/home/dyf/data_generate/persona-instruct/data/lima/epoch/diff/test.jsonl', "r")]
    seed_tasks = seed_tasks
    documents = []
    for tmp in seed_tasks:
        documents.append(tmp['conversations'][0])
    if args.is_vllm == True:
        chat_formatting_function = dynamic_import_function("templates.create_prompt_with_huggingface_tokenizer_template")
        model = vllm.LLM(
            model=model_id,
            tokenizer=model_id,
            tokenizer_mode="auto",
            tensor_parallel_size=torch.cuda.device_count(),
            tokenizer_revision=None, 
            revision=None,
        )
        
        sampling_params = vllm.SamplingParams(
            temperature=0.0,  # greedy decoding
            max_tokens=5000,
            # stop=args.additional_stop_sequence,
            # --additional_stop_sequence',
            # type=str,
            # nargs="+",
            # default=[],
        )
    # log2, documents = main_diff(roundi, seed_tasks, args.is_vllm, args.batch_length, model, sampling_params, chat_formatting_function)
    # log1 = main_com(roundi, seed_tasks, args.is_vllm, model, sampling_params, chat_formatting_function, documents)


    seed_tasks, documents = main_diff(roundi, seed_tasks, args.is_vllm, args.batch_length, model, sampling_params, chat_formatting_function)
    # for roundi in range(2):
    # # roundi = 1
    #     seed_tasks = main_com(roundi, seed_tasks, args.is_vllm, model, sampling_params, chat_formatting_function, documents)

    # output_log_jsonl(os.path.join("/home/dyf/data_generate/persona-instruct/data/lima/final/", f"final.jsonl"), seed_tasks)
