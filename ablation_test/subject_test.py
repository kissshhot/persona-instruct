import json
import spacy
def extract_before_first_comma_or_period(s):
    # 使用逗号或句号分割字符串
    parts = s.split(',', 1)  # 只分割一次
    before_comma = parts[0]  # 逗号前的部分
    
    # 检查句号，如果存在则再分割句号
    before_period = before_comma.split('.', 1)[0]  # 句号前的部分
    return before_period.strip()  # 去除首尾空格

seed_tasks_path = "/home/dyf/data_generate/persona-instruct/data/lima/epoch/diff/test.jsonl"
seed_tasks = [json.loads(l) for l in open(seed_tasks_path, "r")]
persona_doc = []
for tmp in seed_tasks:
    first = extract_before_first_comma_or_period(tmp['questioner'])
    persona_doc.append(first)

# 加载英语模型
nlp = spacy.load("en_core_web_sm")

# 示例字符串列表
string_list = persona_doc[:100]
# string_list = [string_list[0]]
def extract_nouns(strings):
    nouns = set()  # 使用集合以避免重复
    for sentence in strings:
        doc = nlp(sentence)
        for token in doc:
            if token.pos_ == "NOUN":  # 检查词性是否为名词
                nouns.add(token.text.lower())  # 提取名词并添加到集合，转换为小写以避免重复
    return nouns

# 提取名词
unique_nouns = extract_nouns(string_list)

# 输出结果
print("提取的名词：", unique_nouns)
print("名词数量：", len(unique_nouns))