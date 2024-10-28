persona_generate_simple = '''
### text:
{dialogue}

Please create a detailed and high quality description of the person who is most likely to write this text.
'''

persona_generate = '''
I will give you a text in a dataset, and I am tagging this dataset with the most relevant persons and a detailed description of the persons, e.g. the person who is most likely to say|listen|write|read|like|dislike this text, please help me generate a high quality example for this dataset.

For this task you will generate a good length answer using your best helpfulness and wisdom, and No need to include verbose or extraneous information.

Before I give you the text, I'll give you two examples:
Example 1:
### text:
What are the key considerations for scheduling and logistics when hosting a multi-show festival at a performing arts center like the Broward Center?
### questioner:
an event planner or festival organizer with experience in coordinating arts events. They are seeking expert insights to improve their logistical planning, focusing on aspects such as timing, resource allocation, and audience flow. This individual values efficiency and effectiveness in creating a memorable festival experience.
### respondent:
an experienced event management professional specializing in the performing arts. They possess a deep understanding of scheduling challenges and logistical requirements specific to multi-show events. Their expertise includes audience engagement, venue management, and operational strategies, making them well-equipped to provide valuable recommendations.
Example 2:
### text:
Compare and contrast the distribution of public services such as libraries, community centers, and public transportation in different neighborhoods of Halifax, and discuss how the municipal government's urban planning strategies impact access to these services for residents of varying socioeconomic backgrounds.
### questioner:
a researcher, urban planner, or student interested in social equity and urban development. They seek to understand how public service distribution varies across neighborhoods in Halifax and the implications for different socioeconomic groups. This person values data-driven analysis and comprehensive comparisons.
### respondent:
an urban studies expert with a strong background in social policy and community development. They are familiar with Halifax's urban landscape and municipal strategies, particularly how these influence access to public services. Their insights often incorporate statistical analysis, community feedback, and best practices in urban planning.

Now, it is your turn! Given the guidelines and examples above, please create a detailed and high quality description of the person who is most likely to say|listen|write|read|like|dislike this text:
### text:
{dialogue}

Your answer should be as follows:
### questioner:
<a detailed description of the questioner>
### respondent:
<a detailed description of the respondent>
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

persona_com_instruct_generate_rewrite = '''
Please rewrite one new description based on the existing description and the given task or
question. Then, using the new description, create a new, high quality and more challenging version of the task
or question.
### Important: The new description should differ from the previous description and relate to the context
of the question.
### Format:
[Questioner]: Here are the existing description of questioner.
[Respondent]: Here are the existing description of respondent.
[Original Question]: Here is the original question.
Output:
[New Questioner]: Here is the description of new questioner.
[New Respondent]: Here is the description of new respondent.
[New Question]: Here is the new question.
### Your Task:
[Questioner]: {questioner}
[Respondent]: {respondent}
[Original Question]: {question}
Output:
'''
# The topic of the new questioner, respondent and query must differ from the topic of the examples provided.
# The new questioner, respondent and query must differ from the examples provided.
# mustdifferfromtheexamplesprovided.
# 2. Ensure that the new questioner, respondent and query are not associated with any of the examples.
persona_diff_instruct_generate='''
Generate a new questioner, a new respondent and a new corresponding query based on the following questioner, respondent and query.
### Important:
1. The new questioner, respondent and query must differ from the examples provided.
2. Ensure that the content you generate is of high quality.
3. Don't provide a solution or answer to the query.
### Example:
Example 1:
[questioner]: {questioner1}
[respondent]: {respondent1}
[question]: {question1}
Example 2:
[questioner]: {questioner2}
[respondent]: {respondent2}
[question]: {question2}
Example 3:
[questioner]: {questioner3}
[respondent]: {respondent3}
[question]: {question3}
Example 4:
[questioner]: {questioner4}
[respondent]: {respondent4}
[question]: {question4}

Your output should be as follows:
[New Questioner]: Here is the description of new questioner.
[New Respondent]: Here is the description of new respondent.
[New Question]: Here is the new question.
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

answer_generate = '''
Below is an instruction that describes a task, write a response that appropriately completes
the request.
### Instruction:
{instruction}
### Response:
'''

answer_generate_persona = '''
Below is an instruction that describes a task, I want you act as {respondent}, write a response that appropriately completes
the request.
### Instruction:
{instruction}
### Response:
'''