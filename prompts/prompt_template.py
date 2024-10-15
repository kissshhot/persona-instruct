persona_generate = '''
I will give you a dialogue in a dataset, and I am tagging this dataset with the most relevant person and a detailed description of that person, e.g. the person who is most likely to say|listen|write|read|like|dislike this dialogue, please help me generate a high quality example for this dataset.

For this task you will generate a good length answer using your best helpfulness and wisdom.

### dialogue:
{dialogue}

Your answer should be as follows:
description: <a detailed description of the person>
reason: <reason for the person>
'''

instruct_generate = '''

'''

answer_generate = '''

'''