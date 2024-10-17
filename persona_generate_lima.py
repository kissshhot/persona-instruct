from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os
import json
import random
import re
import string
import tqdm
import argparse
from prompts.prompt_template import persona_generate, persona_generate_simple
# os.environ["CUDA_VISIBLE_DEVICES"] = "5,6,7"
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
        default="/home/dyf/data_generate/persona-instruct/data/lima_train.jsonl",
        help="The path to the human written data.",
    )
    parser.add_argument(
        "--use_clf_seed_tasks_only",
        action="store_true",
        help="If specified, we will only use the classification seed tasks to prompt new instructions. This will lead to more classification instructions.",
    )
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    seed_tasks = [json.loads(l) for l in open(args.seed_tasks_path, "r")]
    if args.use_clf_seed_tasks_only:
        seed_tasks = [t for t in seed_tasks if t["is_classification"]]
    os.makedirs(args.batch_dir, exist_ok=True)

    persona_add = []
    all_logs=[]
    for idx in range(len(seed_tasks)): #len(seed_tasks)
        dialogue = ''
        for tx in seed_tasks[idx]['conversations']:
            dialogue += tx + '\n'
        # dialogue = seed_tasks[idx]['conversations'] # + '\n' + seed_tasks[idx]['instances'][0]['input'] + '\n' + seed_tasks[idx]['instances'][0]['output']
        inputs = persona_generate.format(dialogue=dialogue)
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
        result = tokenizer.decode(outputs[0][len(inputs[0]):], skip_special_tokens=True)
        print(tokenizer.decode(outputs[0][len(inputs[0]):], skip_special_tokens=True))
        t = seed_tasks[idx]
        if len(result.split('### persona:\n')) == 2:
            t['persona'] = result.split('### persona:\n')[1]
            all_logs.append(t)
        else:
            t['persona'] = 'error'
            all_logs.append(t)
        # output log at each iteration
    output_log_jsonl(os.path.join(args.batch_dir, "persona_add_lima.jsonl"), all_logs) 