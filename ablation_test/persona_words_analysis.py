import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk import pos_tag
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import json
# 确保已经下载了NLTK的资源
# nltk.download('punkt')
# nltk.download('punkt_tab')
# nltk.download('averaged_perceptron_tagger')
# nltk.download('wordnet2022')
seed_tasks_path = "/home/dyf/data_generate/persona-instruct/data/lima/merged/diff_merged_instruct_20000_person2_round_0.jsonl"
seed_tasks = [json.loads(l) for l in open(seed_tasks_path, "r")]
persona_doc = []
for tmp in seed_tasks:
    persona_doc.append(tmp['questioner'])
# 示例字符串列表
strings = persona_doc

# 准备一个空列表来存储提取的名词
nouns = []

# 遍历字符串列表
for sentence in strings:
    # 提取第一个逗号或句号之前的内容
    if '.' in sentence:
        part = sentence.split(',', 1)[0]
    elif ',' in sentence:
        part = sentence.split('.', 1)[0]
    else:
        part = sentence

    # 分词
    words = word_tokenize(part)

    # 词性标注
    tagged = pos_tag(words)

    # 提取名词
    for word, tag in tagged:
        if tag.startswith('NN'):
            nouns.append(word.lower())

# 创建词云对象
wordcloud = WordCloud(width=1600, height=800, background_color='white').generate(' '.join(nouns))

# 显示词云图
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')  # 不显示坐标轴
plt.show()
plt.savefig(f"/home/dyf/data_generate/persona-instruct/ablation_test/persona_words_analysis.png", format='png', dpi=300)