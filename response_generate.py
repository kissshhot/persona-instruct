# 这里会让模型对指令生成多个回答，并采用rejection sampling 和 preference-based sampling来筛选出一个最好的回答

# rejection sampling
# 接下来，我们设定一些标准，比如：

# 回答必须与问题相关。
# 回答不能含糊或模棱两可。

# preference-based sampling
# 假设我们从过去的用户反馈中了解到用户更喜欢直接、确定性的回答，而不是模棱两可的回答。