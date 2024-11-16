from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel
import pdb
import os
import json
import torch
os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6,7"
tokenizer_embedding = AutoTokenizer.from_pretrained('BAAI/bge-small-en-v1.5')
model_embedding = AutoModel.from_pretrained('BAAI/bge-small-en-v1.5', device_map='auto') # , device_map={"": "cuda"} {"": "cuda"}
model_embedding.eval()
task2 = [json.loads(l) for l in open('/home/dyf/data_generate/persona-instruct/data/lima/epoch/diff/diff_new_instruct_12000_person2_round_0.jsonl', "r")]
txt = []
for task in task2:
    txt.append(task['questioner'])
# Tokenize sentences
encoded_input = tokenizer_embedding(txt, padding=True, truncation=True, return_tensors='pt').to('cuda')
# for s2p(short query to long passage) retrieval task, add an instruction to query (not add instruction for passages)
# encoded_input = tokenizer([instruction + q for q in queries], padding=True, truncation=True, return_tensors='pt')

# Compute token embeddings
with torch.no_grad():
    model_output = model_embedding(**encoded_input)
    # Perform pooling. In this case, cls pooling.
    txt_embeddings = model_output[0][:, 0]
# normalize embeddings
txt_embeddings = torch.nn.functional.normalize(txt_embeddings, p=2, dim=1).to('cpu')
# score_list =[txt_embeddings[0] @ sentence_embedding[i] for i in range(0, len(sentence_embedding))]
tensor1 = torch.load('/home/dyf/data_generate/persona-instruct/embedding/questioner_embedding.pt')
sentence_embedding = torch.cat((tensor1, txt_embeddings), dim=0)
torch.save(sentence_embedding, '/home/dyf/data_generate/persona-instruct/embedding/questioner_embedding1.pt')