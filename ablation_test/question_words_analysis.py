import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk import pos_tag
from collections import Counter, defaultdict
from pyecharts.charts import Pie, Sunburst
from pyecharts import options as opts
import json
# 确保已经下载了NLTK的资源
# nltk.download('punkt')
# nltk.download('averaged_perceptron_tagger')
seed_tasks_path = "/home/dyf/data_generate/persona-instruct/data/lima/merged/diff_merged_instruct_20000_person2_round_0.jsonl"
seed_tasks = [json.loads(l) for l in open(seed_tasks_path, "r")]
persona_doc = []
for tmp in seed_tasks:
    persona_doc.append(tmp['conversations'][0])
# 示例字符串列表
strings = persona_doc
# 示例字符串列表

# 准备一个空字典来存储动词-名词对及其频率
verb_noun_pairs = defaultdict(Counter)

# 定义be动词的所有形式
be_verbs = {"is", "am", "are", "was", "were", "be", "being", "been", "\'m", "\'ve"}

# 遍历字符串列表
for sentence in strings:
    # 分词
    words = word_tokenize(sentence)

    # 词性标注
    tagged = pos_tag(words)

    # 遍历标注后的词
    for i in range(len(tagged)):
        word, tag = tagged[i]
        # 检查是否为动词且不是be动词
        if tag.startswith('VB') and word.lower() not in be_verbs:
            # 查找随后的名词
            for j in range(i+1, len(tagged)):
                next_word, next_tag = tagged[j]
                if next_tag.startswith('NN'):
                    # 构建动词-名词对
                    verb_noun_pairs[word.lower()][next_word.lower()] += 1
                    break

# 统计每个动词的总频率
verb_frequency = Counter()
for verb, nouns in verb_noun_pairs.items():
    verb_frequency[verb] = sum(nouns.values())

# 选择频率前50的动词
top_50_verbs = [verb for verb, _ in verb_frequency.most_common(50)]

# 准备绘制扇形图的数据
top_verbs_data = {verb: sum(nouns.values()) for verb, nouns in verb_noun_pairs.items() if verb in top_50_verbs}

sunburst_data = [
    {
        "name": verb,
        "value": top_verbs_data[verb],
        "label_opts": opts.LabelOpts(
            position="inside",
            font_size=12,  # 内层动词字体大小
            color="black"
        ),
        "children": [
            {
                "name": noun,
                "value": freq,
                "label_opts": opts.LabelOpts(is_show=False)  # 隐藏外层名词标签
            }
            for noun, freq in verb_noun_pairs[verb].items()
        ]
    }
    for verb in top_50_verbs
]

# 创建旭日图
sunburst = Sunburst()
sunburst.add(
    series_name="Top 50 Verbs and Noun Pairs",
    data_pair=sunburst_data,
    radius=[0, "70%", "90%"],  # 控制旭日图的内外半径，增加内层半径
)

# 设置全局配置项，隐藏工具栏并设置标题
sunburst.set_global_opts(
    title_opts=opts.TitleOpts(title="Top 50 Verbs and Noun Pairs Distribution"),
    toolbox_opts=opts.ToolboxOpts(is_show=False),
    legend_opts=opts.LegendOpts(is_show=False)
)


# 渲染图表到HTML文件
sunburst.render("/home/dyf/data_generate/persona-instruct/ablation_test/question_words_analysis.html")