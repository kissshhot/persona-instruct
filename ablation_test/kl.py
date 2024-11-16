import numpy as np
import torch
import json
from transformers import AutoModel, AutoTokenizer
from scipy.stats import entropy
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
tokenizer_embedding = AutoTokenizer.from_pretrained('BAAI/bge-small-en-v1.5')
model_embedding = AutoModel.from_pretrained('BAAI/bge-small-en-v1.5') # , device_map={"": "cuda"}
model_embedding.eval()
seed_tasks_path = "/home/dyf/data_generate/persona-instruct/data/lima/epoch/diff/diff_new_instruct_20000_person2_round_0.jsonl"
seed_tasks = [json.loads(l) for l in open(seed_tasks_path, "r")]
persona_doc = []
# for tmp in seed_tasks:
#     persona_doc.append(tmp['questioner'])
for tmp in seed_tasks:
    persona_doc.append(tmp['conversations'][0])

# Tokenize sentences
encoded_input = tokenizer_embedding(persona_doc[:2000], padding=True, truncation=True, return_tensors='pt')# .to('cuda')
# for s2p(short query to long passage) retrieval task, add an instruction to query (not add instruction for passages)
# encoded_input = tokenizer([instruction + q for q in queries], padding=True, truncation=True, return_tensors='pt')

# Compute token embeddings
with torch.no_grad():
    model_output = model_embedding(**encoded_input)
    # Perform pooling. In this case, cls pooling.
    txt_embeddings = model_output[0][:, 0]
# normalize embeddings

# kl散度是计算分布的，应该不能归一化
# txt_embeddings = torch.nn.functional.normalize(txt_embeddings, p=2, dim=1)



# 假设 embeddings 是 2000条数据的嵌入矩阵
# 分割成前500条和后1500条
result = []
for i in range(0, len(txt_embeddings), 1000):
    try:
        embeddings_500 = txt_embeddings[i:i+1000]
        embeddings_1500 = txt_embeddings[i+1000:i+2000]
    except:
        break
    # 降维到较低维度（例如2D）
    pca = PCA(n_components=2)
    reduced_500 = pca.fit_transform(embeddings_500)
    reduced_1500 = pca.fit_transform(embeddings_1500)

    # 计算直方图，得到概率分布的近似
    # 使用相同的bin区间，bins参数可根据数据情况调整
    hist_500, _ = np.histogramdd(reduced_500, bins=30, density=True)
    hist_1500, _ = np.histogramdd(reduced_1500, bins=30, density=True)

    # 假设 hist_500 是通过 np.histogramdd 获得的二维直方图数据
    hist_500, edges = np.histogramdd(reduced_500, bins=30, density=True)

    # 绘制热力图
    plt.figure(figsize=(8, 6))
    plt.imshow(hist_500.T, origin='lower', cmap='Blues', extent=[edges[0][0], edges[0][-1], edges[1][0], edges[1][-1]])
    plt.colorbar(label='Density')  # 添加颜色条表示密度
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.title('2D Histogram Heatmap')
    plt.show()
    plt.savefig(f"/home/dyf/data_generate/persona-instruct/ablation_test/kl_figure/persona/histogram_{i}.png", format='png', dpi=300)
    
    hist_1500, edges = np.histogramdd(reduced_1500, bins=30, density=True)

    # 绘制热力图
    plt.figure(figsize=(8, 6))
    plt.imshow(hist_1500.T, origin='lower', cmap='Blues', extent=[edges[0][0], edges[0][-1], edges[1][0], edges[1][-1]])
    plt.colorbar(label='Density')  # 添加颜色条表示密度
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.title('2D Histogram Heatmap')
    plt.show()
    plt.savefig(f"/home/dyf/data_generate/persona-instruct/ablation_test/kl_figure/persona/histogram_{i+1}.png", format='png', dpi=300)

    # 将直方图结果展平成一维，并确保每个分布的和为1
    hist_500 = hist_500 / np.sum(hist_500)
    hist_1500 = hist_1500 / np.sum(hist_1500)
    print(hist_500.shape, hist_1500.shape)
    # 后面不一定需要
    epsilon = 1e-10  # 小常数
    hist_500_smoothed = hist_500 + epsilon
    hist_1500_smoothed = hist_1500 + epsilon
    # 计算KL散度
    kl_divergence = entropy(hist_500_smoothed, hist_1500_smoothed)
    print("KL散度:", kl_divergence)
    average_kl_divergence = np.mean(kl_divergence)  # 平均值
    result.append(np.around(kl_divergence, decimals=2))
    print("平均KL散度:", average_kl_divergence)

print(result)
# 假设这是你的数据列表
x = range(len(result))  # X轴数据
y = result  # Y轴数据

# 创建一个图形
plt.figure()

# 绘制折线图
plt.plot(x, y, marker='o')  # 'o'表示用圆圈标记每个数据点

# 在每个数据点上标注数值
for i in range(len(x)):
    plt.text(x[i], y[i], f'{y[i]}', ha='center', va='bottom')  # ha和va用于调整文本位置

# 添加标题和标签
plt.title('折线统计图')
plt.xlabel('X轴')
plt.ylabel('Y轴')


# 显示图形
plt.show()
plt.savefig(f"/home/dyf/data_generate/persona-instruct/ablation_test/kl_figure/persona/kl_result.png", format='png', dpi=300)