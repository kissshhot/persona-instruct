from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os
import json
import pdb
import random
import re
import string
import vllm
from importlib import import_module
from tqdm import tqdm
import argparse
from prompts.prompt_template_persona2 import resonpdant_generate, persona_generate_simple
# os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6,7"
model_id = "/data1/dyf/model/Mistral-7B-Instruct-v0.3/"
tokenizer = AutoTokenizer.from_pretrained(model_id)
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

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--batch_dir",
        type=str,
        # required=True,
        default="/home/dyf/data_generate/persona-instruct/data/lima/persona2",
        help="The directory where the batch is stored.",
    )
    parser.add_argument(
        "--seed_tasks_path",
        type=str,
        # required=True,
        default="/home/dyf/data_generate/persona-instruct/data/lima_train.jsonl",
        help="The path to the human written data.",
    )
    parser.add_argument(
        "--use_clf_seed_tasks_only",
        action="store_true",
        help="If specified, we will only use the classification seed tasks to prompt new instructions. This will lead to more classification instructions.",
    )
    parser.add_argument(
        "--use_vllm",
        # required=True,
        action="store_true",
        # help="The path to the human written data.",
    )
    return parser.parse_args()

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

if __name__ == "__main__":
    args = parse_args()

    seed_tasks = [json.loads(l) for l in open(args.seed_tasks_path, "r")]
    if args.use_clf_seed_tasks_only:
        seed_tasks = [t for t in seed_tasks if t["is_classification"]]
    os.makedirs(args.batch_dir, exist_ok=True)

    persona_add = []
    all_logs=[]
    if args.use_vllm == True:
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
            # top_p=0.9,
            max_tokens=5000,
            # stop=args.additional_stop_sequence,
            # --additional_stop_sequence',
            # type=str,
            # nargs="+",
            # default=[],
        )
        for idx in tqdm(range(len(seed_tasks))): #len(seed_tasks)
            questioner = seed_tasks[idx]['questioner']
            question = seed_tasks[idx]['conversations'][0]
            # dialogue = seed_tasks[idx]['conversations'] # + '\n' + seed_tasks[idx]['instances'][0]['input'] + '\n' + seed_tasks[idx]['instances'][0]['output']
            prompt = resonpdant_generate.format(questioner=questioner, question=question).strip()
            # while True:
            result = use_vllm([prompt], model, sampling_params, chat_formatting_function).strip()
            try:
                respondent = result.split('### respondent:\n')[1].strip()
                # if len(respondent.split('\n')) >= 2:
                #     respondent = respondent.split('\n')[0]
                # break
            except:
                #pdb.set_trace()
                continue
            print(result)
            t = seed_tasks[idx]
            t['respondent'] = respondent
            all_logs.append(t)
            print(t['respondent'])
            if len(all_logs) >=2:
                if all_logs[-1] == all_logs[-2]:
                    pdb.set_trace()
            # output log at each iteration
            output_log_jsonl(os.path.join(args.batch_dir, "respondant_add_w_vllm.jsonl"), all_logs)
    else:
        # tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")
        for idx in tqdm(range(len(seed_tasks))): #len(seed_tasks)
            # dialogue = ''
            # for tx in seed_tasks[idx]['conversations']:
            #     dialogue += tx + '\n'
            dialogue = seed_tasks[idx]['conversations'][0]
            # dialogue = seed_tasks[idx]['conversations'] # + '\n' + seed_tasks[idx]['instances'][0]['input'] + '\n' + seed_tasks[idx]['instances'][0]['output']
            prompt = persona_generate.format(dialogue=dialogue)
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
            while True:
                outputs = model.generate(inputs, max_new_tokens=5000, do_sample=True, temperature=0.7, top_p=0.9)
                result = tokenizer.decode(outputs[0][len(inputs[0]):], skip_special_tokens=True)
                try:
                    questioner = result.split('### questioner:\n')[1].split('\n')[0]
                    respondent = result.split('### respondent:\n')[1]
                    if len(respondent.split('\n')) >= 2:
                        respondent = respondent.split('\n')[0]
                    break
                except:
                    pdb.set_trace()
                    continue
            print(result)
            t = seed_tasks[idx]
            t['questioner'] = questioner
            t['respondent'] = respondent
            all_logs.append(t)
            print(t['questioner'])
            print(t['respondent'])
            if all_logs[-1] == all_logs[-2]:
                pdb.set_trace()
            # output log at each iteration
            output_log_jsonl(os.path.join(args.batch_dir, "persona_add_lima_persona2_wo_vllm.jsonl"), all_logs)