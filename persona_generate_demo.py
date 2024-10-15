from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os
import json
import random
import re
import string
import tqdm
import argparse
from prompts.prompt_template import persona_generate
#os.environ["CUDA_VISIBLE_DEVICES"] = "5,6,7"
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
        default="/home/dyf/data_generate/our_demo/data",
        help="The directory where the batch is stored.",
    )
    parser.add_argument(
        "--seed_tasks_path",
        type=str,
        # required=True,
        default="/home/dyf/data_generate/our_demo/data/seed_tasks.jsonl",
        help="The path to the human written data.",
    )
    parser.add_argument(
        "--num_instructions_to_generate",
        type=int,
        default=100,
        help="th",
    )
    parser.add_argument(
        "--use_clf_seed_tasks_only",
        action="store_true",
        help="If specified, we will only use the classification seed tasks to prompt new instructions. This will lead to more classification instructions.",
    )
    parser.add_argument(
        "--engine",
        type=str,
        default="davinci",
        help="The engine to use."
    )
    parser.add_argument(
        "--num_prompt_instructions",
        type=int,
        default=8,
        help="The number of instructions to use in the prompt."
    )
    parser.add_argument(
        "--request_batch_size",
        type=int,
        default=1,
        help="The number of requests to send to GPT3 at a time."
    )
    parser.add_argument(
        "--api_key",
        type=str,
        default="sk-cBMDIQFHqCyqrGlSFb26F27dD9C4445d971a0673A73c7f9a",
        help="The API key to use. If not specified, the key will be read from the environment variable OPENAI_API_KEY."
    )
    parser.add_argument(
        "--organization",
        type=str,
        help="The organization to use. If not specified, the default organization id will be used."
    )
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    seed_tasks = [json.loads(l) for l in open(args.seed_tasks_path, "r")]
    if args.use_clf_seed_tasks_only:
        seed_tasks = [t for t in seed_tasks if t["is_classification"]]
    os.makedirs(args.batch_dir, exist_ok=True)
    request_idx = 0
    persona_add = []
    all_logs=[]
    for idx in range(len(seed_tasks)):
        task = seed_tasks[idx]['']
        inputs = persona_generate.format(task=task)
        conversation = [{"role": "user", "content": inputs}]
        # tools = [get_current_weather]


        # format and tokenize the tool use prompt 
        inputs = tokenizer.apply_chat_template(
                    conversation,
                    add_generation_prompt=True,
                    # return_dict=True,
                    return_tensors="pt",
        )

        inputs.to(model.device)
        outputs = model.generate(inputs, max_new_tokens=5000)
        result = tokenizer.decode(outputs[0][len(inputs[0]):], skip_special_tokens=True)
        print(tokenizer.decode(outputs[0][len(inputs[0]):], skip_special_tokens=True))
        t = seed_tasks[idx]
        t['persona'] = result.split('reason')[0]
        all_logs.append(t)
        # output log at each iteration
    output_log_jsonl(os.path.join(args.batch_dir, "persona_add.jsonl"), all_logs) 