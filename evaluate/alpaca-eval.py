import datasets
from alpaca_eval import evaluate
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os
import json
from tqdm import tqdm
from accelerate import Accelerator
os.environ["CUDA_VISIBLE_DEVICES"] = "5,6,7"
model_id = "/data1/dyf/model/Mistral-7B-Instruct-v0.3/"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")
def generate(instruction):
    accelerator = Accelerator()
    models = accelerator.prepare(model)
    conversation = [{"role": "user", "content": instruction}]
    inputs = tokenizer.apply_chat_template(
                conversation,
                add_generation_prompt=True,
                # return_dict=True,
                return_tensors="pt",
    )

    inputs = inputs.to('cuda')
    with torch.no_grad():
        outputs = model(inputs, max_new_tokens=5000, do_sample=False)
    # outputs = model.generate(inputs, max_new_tokens=5000, do_sample=False)
    result = tokenizer.decode(outputs[0][len(inputs[0]):], skip_special_tokens=True)
    print(result)
    return result

if __name__ == "__main__":
    # eval_set = datasets.load_dataset("tatsu-lab/alpaca_eval", "alpaca_eval", force_download=True, resume_download=False)["eval"]
    filepath = '/home/dyf/data_generate/persona-instruct/evaluate/data/alpaca_eval.json'
    my_model = 'Mistral-7B-Instruct-v0.3'
    outputpath = f'/home/dyf/data_generate/persona-instruct/evaluate/outputs/alpaca_{my_model}.json'
    outputs = []
    with open(filepath, encoding="utf-8") as f:
        eval_set = json.load(f)
    for example in tqdm(eval_set):
        # generate here is a placeholder for your models generations
        example["output"] = generate(example["instruction"])
        example["generator"] = my_model # name of your model
        outputs.append(example)
    with open(outputpath, 'w') as json_file:
        json.dump(outputs, json_file)
    
    # # 假设你有一个模型输出文件
    # model_outputs_file = 'path/to/model_outputs.json'
    
    # # 进行评估
    # results = evaluate(model_outputs_file)
    # print(results)