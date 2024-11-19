persona_generate_simple = '''
### text:
{dialogue}

Please create a detailed and high quality description of the person who is most likely to write this text.
'''

# persona_generate = '''
# I will give you a text in a dataset, and I am tagging this dataset with the most relevant persons and a detailed description of the persons, e.g. the person who is most likely to say|listen|write|read|like|dislike this text, please help me generate a high quality example for this dataset.

# For this task you will generate a good length answer using your best helpfulness and wisdom, and No need to include verbose or extraneous information.

# Before I give you the text, I'll give you two examples:
# Example 1:
# ### text:
# What are the key considerations for scheduling and logistics when hosting a multi-show festival at a performing arts center like the Broward Center?
# ### questioner:
# an event planner or festival organizer with experience in coordinating arts events. They are seeking expert insights to improve their logistical planning, focusing on aspects such as timing, resource allocation, and audience flow. This individual values efficiency and effectiveness in creating a memorable festival experience.
# ### respondent:
# an experienced event management professional specializing in the performing arts. They possess a deep understanding of scheduling challenges and logistical requirements specific to multi-show events. Their expertise includes audience engagement, venue management, and operational strategies, making them well-equipped to provide valuable recommendations.
# Example 2:
# ### text:
# Compare and contrast the distribution of public services such as libraries, community centers, and public transportation in different neighborhoods of Halifax, and discuss how the municipal government's urban planning strategies impact access to these services for residents of varying socioeconomic backgrounds.
# ### questioner:
# a researcher, urban planner, or student interested in social equity and urban development. They seek to understand how public service distribution varies across neighborhoods in Halifax and the implications for different socioeconomic groups. This person values data-driven analysis and comprehensive comparisons.
# ### respondent:
# an urban studies expert with a strong background in social policy and community development. They are familiar with Halifax's urban landscape and municipal strategies, particularly how these influence access to public services. Their insights often incorporate statistical analysis, community feedback, and best practices in urban planning.

# Now, it is your turn! Given the guidelines and examples above, please create a detailed and high quality description of the person who is most likely to say|listen|write|read|like|dislike this text:
# ### text:
# {dialogue}

# Your answer should be as follows:
# ### questioner:
# <a detailed description of the questioner>
# ### respondent:
# <a detailed description of the respondent>
# '''

persona_generate = '''
I will give you a text in a dataset, and I am tagging this dataset with the most relevant person and a detailed description of the person, e.g. the person who is most likely to say|listen|write|read this text, please help me generate a high quality example for this dataset.

For this task you will generate a good length answer using your best helpfulness and wisdom, and No need to include verbose or extraneous information.

Before I give you the text, I'll give you two examples:
Example 1:
### text:
What are the key considerations for scheduling and logistics when hosting a multi-show festival at a performing arts center like the Broward Center?
### questioner:
an event planner or festival organizer with experience in coordinating arts events. They are seeking expert insights to improve their logistical planning, focusing on aspects such as timing, resource allocation, and audience flow. This individual values efficiency and effectiveness in creating a memorable festival experience.
Example 2:
### text:
Compare and contrast the distribution of public services such as libraries, community centers, and public transportation in different neighborhoods of Halifax, and discuss how the municipal government's urban planning strategies impact access to these services for residents of varying socioeconomic backgrounds.
### questioner:
a researcher, urban planner, or student interested in social equity and urban development. They seek to understand how public service distribution varies across neighborhoods in Halifax and the implications for different socioeconomic groups. This person values data-driven analysis and comprehensive comparisons.

Now, it is your turn! Given the guidelines and examples above, please create a detailed and high quality description of the person who is most likely to say|listen|write|read this text:
### text:
{dialogue}

Your answer should be as follows:
### questioner:
<a detailed description of the questioner>
'''
# description: <a detailed description of the person>
# reason: <reason for the person>
persona_com_instruct_generate = '''
Please generate **only one** new description based on the existing description and the given task or
question. Then, using all the descriptions, create a new, more challenging version of the task
or question.
### Important: The new description should differ from the previous description and relate to the context
of the question.
### Format:
[questioner]: Here are the existing description of questioner.
[respondent]: Here are the existing description of respondent.
[Original Question]: Here is the original question.
Output:
[New questioner]: Here is the new description of questioner.
[New respondent]: Here is the new description of respondent.
[New Question]: Here is the new question.
### Your Task:
[questioner]: {questioner}
[respondent]: {respondent}
[Original Question]: {question}
Output:
'''

# Please generate one new questioner and respondent based on the existing questioner, respondent and the given question. Then, using the new questioner, generate a new, high quality and more challenging version of the question.
# ### Important:
# 1. The new questioner should differ from the previous questioner and relate to the context of the new question.
# 2. You need to explain why the new question is more challenging.
# 3. Don't provide a solution or answer to the new question.
# ### Format:
# [Questioner]: Here is the existing description of questioner.
# [Original Question]: Here is the original question.
# [Respondent]: Here is the existing description of respondent.
# Output:
# [New Questioner]: Here is the description of new questioner.
# [New Question]: Here is the new question.
# [New Respondent]: Here is the description of new respondent.
# [Reason]: Your reason for the new question.
# ### Your Task:
# [Questioner]: {questioner}
# [Original Question]: {question}
# [Respondent]: {respondent}
# Output:
# 1. The new questioner should differ from the previous questioner and relate to the context of the new question.

persona_com_instruct_generate_rewrite = '''
Please generate one new questioner based on the existing questioner and the given question. Then, using the new questioner, generate a new, high quality and more challenging version of the question.
### Important:
1. The new questioner should differ from the previous questioner and relate to the context of the new question.
2. When generating new questions, I want you to play the role of the new questioner and generate a new question that you are most likely to say|listen|write|read.
3. You need to explain why the new question is more challenging.
4. Don't provide a solution or answer to the new question.
### Format:
[Questioner]: Here is the existing description of questioner.
[Original Question]: Here is the original question.
Output:
[New Questioner]: Here is the description of new questioner.
[New Question]: Here is the new question.
[Reason]: Your reason for the new question.
### Your Task:
[Questioner]: {questioner}
[Original Question]: {question}
Output:
'''

persona_com_instruct_generate_rewrite_wo_persona = '''
Please generate a new, high quality and more challenging version of the question based on the given question.
### Important:
1. You need to explain why the new question is more challenging.
2. Don't provide a solution or answer to the new question.
### Format:
[Original Question]: Here is the original question.
Output:
[New Question]: Here is the new question.
[Reason]: Your reason for the new question.
### Your Task:
[Original Question]: {question}
Output:
'''

# The topic of the new questioner, respondent and query must differ from the topic of the examples provided.
# The new questioner, respondent and query must differ from the examples provided.
# mustdifferfromtheexamplesprovided.
# 2. Ensure that the new questioner, respondent and query are not associated with any of the examples.
# 回答者不会知道关于问题者的上下文信息，所以新问题应当清晰明了
# 这个新问题将会被用于对另外一个大模型进行提问，在提问时仅提供问题不会提供问题者的信息
# 新的问题者后续可能会被另外一个大模型改写，改写时不会提供样例中的信息，所以新的问题者请不要出现Example 1等省略信息的字样
# in work, study or life
# 4. The new question will be used to ask another large language model, and only the question will be provided without any information about the questioner.
# 5. The new questioner may later be rewritten by another large language model. During the rewrite, the information in the example will not be provided, so the new questioner should avoid using abbreviations like "Example 1" or similar placeholder references.

# Generate a new questioner, a new query and a new respondent based on the following examples.
# ### Example:
# Example 1:
# [questioner]: {questioner1}
# [question]: {question1}
# [respondent]: {respondent1}
# Example 2:
# [questioner]: {questioner2}
# [question]: {question2}
# [respondent]: {respondent2}
# ### Important:
# 1. The new questioner and the Example questioners are in different domains.
# 2. The new questioner must have a collaborative relationship with both Example 1 and Example 2 questioners.
# 3. You need to explain the collaborative relationship between the new questioner and the example questioners.
# 4. The new question you generate and the example questions are independent of each other.
# 5. The new question will be used to ask another large language model, and only the question will be provided without any information about the questioner.
# 6. The new questioner may later be rewritten by another large language model. During the rewrite, the information in the example will not be provided, so the new questioner should avoid using abbreviations like "Example 1" or similar placeholder references.
# 7. Don't provide a solution or answer to the new query.

# Your output should be as follows:
# [New Questioner]: Here is the description of new questioner.
# [New Question]: Here is the new question.
# [Collaborative Relationship]: Here is the collaborative relationship between the new questioner and the example questioners.

# Generate a new questioner, a new query and a new respondent based on the following examples.
# ### Example:
# Example 1:
# [questioner]: {questioner1}
# [question]: {question1}
# [respondent]: {respondent1}
# Example 2:
# [questioner]: {questioner2}
# [question]: {question2}
# [respondent]: {respondent2}
# ### Important:
# 1. The new questioner and the Example questioners are in different domains.
# 2. The new questioner must have a collaborative relationship with both Example 1 and Example 2 questioners.
# 3. You need to explain the collaborative relationship between the new questioner and the example questioners.
# 4. The new question you generate and the example questions are independent of each other.
# 5. The new question will be used to ask another large language model, and only the question will be provided without any information about the questioner.
# 6. The new questioner may later be rewritten by another large language model. During the rewrite, the information in the example will not be provided, so the new questioner should avoid using abbreviations like "Example 1" or similar placeholder references.
# 7. Don't provide a solution or answer to the new query.

# Your output should be as follows:
# [New Questioner]: Here is the description of new questioner.
# [New Question]: Here is the new question.
# [New Respondent]: Here is the description of new respondent.
# [Collaborative Relationship]: Here is the collaborative relationship between the new questioner and the example questioners.

# persona_diff_instruct_generate='''
# Generate a new questioner and a new query based on the following examples.
# ### Example:
# Example 1:
# [questioner]: {questioner1}
# [question]: {question1}
# Example 2:
# [questioner]: {questioner2}
# [question]: {question2}
# ### Important:
# 1. The new questioner and the Example questioners are in different domains.
# 2. The new questioner must have a collaborative relationship with both Example 1 and Example 2 questioners.
# 3. You need to explain the collaborative relationship between the new questioner and the example questioners.
# 4. The new question you generate and the example questions are independent of each other.
# 5. The new question will be used to ask another large language model, and only the question will be provided without any information about the questioner.
# 6. The new questioner may later be rewritten by another large language model. During the rewrite, the information in the example will not be provided, so the new questioner should avoid using abbreviations like "Example 1" or similar placeholder references.
# 7. Don't provide a solution or answer to the new query.

# Your output should be as follows:
# [New Questioner]: Here is the description of new questioner.
# [New Question]: Here is the new question.
# [Collaborative Relationship]: Here is the collaborative relationship between the new questioner and the example questioners.
# '''
persona_diff_instruct_generate_re='''
Generate a new questioner, a new respondent and a new query based on the following examples.
### Example:
Example 1:
[questioner]: {questioner1}
[question]: {question1}
[respondent]: {respondent1}
Example 2:
[questioner]: {questioner2}
[question]: {question2}
[respondent]: {respondent2}
### Important:
1. The new questioner must have a collaborative relationship with both Example 1 and Example 2 questioners in work, study or life.
2. You need to explain the collaborative relationship between the new questioner and the example questioners.
3. The new question you generate and the example questions are independent of each other.
4. Don't provide a solution or answer to the new query.

Your output should be as follows:
[New Questioner]: Here is the description of new questioner.
[New Question]: Here is the new question.
[New Respondent]: Here is the description of new respondent.
[Collaborative Relationship]: Here is the collaborative relationship between the new questioner and the example questioners.
'''

# Example 2:
# [questioner]: {questioner2}
# [question]: {question2}
# [respondent]: {respondent1}
# [New Respondent]: Here is the description of new respondent.

# 2. Ensure that the content you generate is highly diverse.
# The new questioner, respondent and query must differ from the examples provided.
# The data you generate must be unrelated to the example provided.
# Example 1:
# Example 2:
# [questioner]: {questioner2}
# [respondent]: {respondent2}
# [question]: {question2}
# Example 3:
# [questioner]: {questioner3}
# [respondent]: {respondent3}
# [question]: {question3}
# Example 4:
# [questioner]: {questioner4}
# [respondent]: {respondent4}
# [question]: {question4}

persona_diff_instruct_generate_simple='''
Generate a new questioner and a new corresponding query based on the following examples.
### Important:
1. The new questioner and query must differ from the examples provided.
2. Ensure that the content you generate is of high quality.
3. Don't provide a solution or answer to the query.
### Example:
Example 1:
[questioner]: {questioner1}
[question]: {question1}
Example 2:
[questioner]: {questioner2}
[question]: {question2}
Example 3:
[questioner]: {questioner3}
[question]: {question3}

Your output should be as follows:
[New Questioner]: Here is the description of new questioner.
[New Question]: Here is the new question.
'''

# Example 4:
# [questioner]: {questioner4}
# [question]: {question4}


persona_diff_instruct_generate='''
Generate a new questioner and a new query based on the following examples.
### Example:
Example 1:
[questioner]: {questioner1}
[question]: {question1}
Example 2:
[questioner]: {questioner2}
[question]: {question2}
### Important:
1. The new questioner and the Example questioners are in different domains.
2. The new questioner must have a collaborative relationship with both Example 1 and Example 2 questioners.
3. You need to explain the collaborative relationship between the new questioner and the example questioners.
4. The new question you generate and the example questions are independent of each other.
5. The new question will be used to ask another large language model, and only the question will be provided without any information about the questioner.
6. The new questioner may later be rewritten by another large language model. During the rewrite, the information in the example will not be provided, so the new questioner should avoid using abbreviations like "Example 1" or similar placeholder references.
7. Don't provide a solution or answer to the new query.

Your output should be as follows:
[New Questioner]: Here is the description of new questioner.
[New Question]: Here is the new question.
[Collaborative Relationship]: Here is the collaborative relationship between the new questioner and the example questioners.
'''
persona_diff_instruct_generate_wo_persona='''
Generate a new query based on the following examples.
### Example:
Example 1:
[question]: {question1}
Example 2:
[question]: {question2}
### Important:
1. The new question you generate and the example questions are independent of each other.
2. Ensure that the new question you generate is of high quality.
3. Don't provide a solution or answer to the query.

Your output should be as follows:
[New Question]: Here is the new question.
'''
# 2. Ensure that the content you generate is of high quality.
# Example 3:
# [question]: {question3}
# Example 4:
# [question]: {question4}

# The new questioners and questions you generate should be able to be used to extend the capabilities of large language models.
# 7. The new questioners and questions you generate should be able to be used to extend the capabilities of large language models.
persona_diff_instruct_generate_wo_question='''
Generate a new questioner and a new query based on the following examples.
### Example:
Example 1:
[questioner]: {questioner1}
[question]: {question1}
Example 2:
[questioner]: {questioner2}
[question]: {question2}
### Important:
1. The new questioner and the Example questioners are in different domains.
2. The new questioner is someone who, in some scenarios, may have a collaborative relationship with the Example 1 and Example 2 questioners.
3. You need to explain the possible collaborative relationship between the new questioner and the Example questioners.
4. You can't refer to the Example questioners in the description of the new questioner, the description of the new questioner should be more general.
5. The new question you generate and the example questions are independent of each other.
6. When generating the new question, I want you to play the role of the new questioner and generate a new question that you are most likely to say|listen|write|read.
7. Ensure that the new question you generate is of high quality.
8. Don't provide a solution or answer to the new query.

Your output should be as follows:
[New Questioner]: Here is the description of the new questioner.
[New Question]: Here is the new question.
[Collaborative Relationship]: Here is the possible collaborative relationship between the new questioner and the example questioners.
'''

persona_diff_instruct_generate_reverse='''
Generate a new questioner and a new query based on the following examples.
### Example:
Example 1:
[questioner]: {questioner1}
[question]: {question1}
Example 2:
[questioner]: {questioner2}
[question]: {question2}
### Important:
1. The new questioner and the Example questioners are in different domains.
2. The new questioner is someone who, in some scenarios, may have a collaborative relationship with the Example 1 and Example 2 questioners.
3. You need to explain the possible collaborative relationship between the new questioner and the Example questioners.
4. You can't refer to the Example questioners in the description of the new questioner, the description of the new questioner should be more general.
5. The new question you generate and the example questions are independent of each other.
6. When generating the new question, I want you to play the role of the new questioner and generate a question that you are most likely to say|listen|write|read.
7. Don't provide a solution or answer to the new query.

Your output should be as follows:
[New Questioner]: Here is the description of new questioner.
[New Question]: Here is the new question.
[Collaborative Relationship]: Here is the possible collaborative relationship between the new questioner and the example questioners.
'''

instruct_generate = '''
I am creating a high quality dataset to test the capabilities of AI in different professions and fields, please help me generate a high quality example for this dataset!

First, I will provide you with some examples that contain a instruction and the most relevant persona for this instruction and his detailed description.

Example 1:
{example1}
Example 2:
{example2}
Example 3:
{example3}
Example 4:
{example4}
Example 5:
{example5}

Now it is your turn! Given the guidelines and examples above, please create a new instruction and a most relevant persona and his detailed description, the example you create can be relevant to any domain but must be of high quality and of a certain level of difficulty to test the capabilities of AI!

Your answer should be as follows:
### persona:
<a detailed description of the person>
### instruction:
<the instruction you create>
'''

resonpdant_generate = '''
I will give you a pair of questioner and question in a dataset, and I am tagging this dataset with the best fit respondant and a detailed description of the respondant, please help me generate a high quality example for this dataset.
For this task you will generate a good length answer using your best helpfulness and wisdom, and No need to include verbose or extraneous information.
Before I give you the text, I'll give you two examples:
Example 1:
### questioner:
A neuroscientist or researcher specializing in neuroanatomy and brain development. They are curious about the intricate workings of the brain and are seeking to expand their understanding of brain cell behavior, particularly in relation to movement and migration. This individual values scientific accuracy and is interested in exploring the boundaries of what is known about the brain.
### question:
Can brain cells move? By movement I mean long distance migration (preferably within the brain only).
### respondant:
An expert in neuroscience, with a focus on cellular migration and brain development. They have a deep understanding of the complex processes that govern brain cell movement, both within and beyond the brain. Their research often involves studying the mechanisms that enable brain cells to migrate and adapt, making them well-equipped to provide insightful responses to questions about this topic.
Example 2:
### questioner:
a computer science student or professional with a focus on hardware design and architecture. They have a strong understanding of RISC and RISC-V processors, and are currently studying the MIPS processor. They are curious about the reasons behind the popularity of CISC architectures, particularly the x86, and are seeking a detailed explanation to broaden their knowledge. This individual values practical applications and efficiency in hardware design.
### question:
In our computer systems lecture we were introduced to the MIPS processor. It was (re)developed over the course of the term and has in fact been quite easy to understand. It uses a RISC design, that is its elementary commands are regularly encoded and there are only few of them in order to keep the wires simple.\nIt was mentioned that CISC follows a different philosophy. I looked briefly at the x86 instruction set and was shocked. I can not image how anyone would want to build a processor that uses so complex a command set!\nSo I figure there have to be good arguments why large portions of the processor market use CISC architectures. What are they?
### respondant:
an experienced computer architect or hardware engineer with expertise in both RISC and CISC architectures. They have a deep understanding of the trade-offs between the two, including performance, complexity, and power consumption. Their insights are grounded in practical applications and real-world examples, making them well-equipped to provide a comprehensive explanation for the popularity of CISC architectures, such as the x86. This individual values evidence-based analysis and the ability to communicate complex concepts clearly.

### Important:
Don't provide a solution or answer to the question.
### Your Task:
### questioner:
{question}
### question:
{questioner}

Your output should be as follows:
### respondant:
<a detailed description of the respondant>
'''

answer_generate = '''
Here is an instruction that describes a task, write a response that appropriately completes the request.
For this task you will generate a good length answer using your best helpfulness and wisdom, and No need to include verbose or extraneous information.

### Instruction:
{instruction}
### Response:
'''

answer_generate_persona = '''
Here is an instruction that describes a task, I want you act as:
### 
{respondant}
###
write a response that appropriately completes the request.
For this task you will generate a good length answer using your best helpfulness and wisdom, and No need to include verbose or extraneous information.

### Instruction:
{instruction}
### Response:
'''