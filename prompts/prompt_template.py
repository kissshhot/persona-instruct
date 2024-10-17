persona_generate_simple = '''
### text:
{dialogue}

Please create a detailed and high quality description of the person who is most likely to write this text.
'''

persona_generate = '''
I will give you a text in a dataset, and I am tagging this dataset with the most relevant person and a detailed description of that person, e.g. the person who is most likely to say|listen|write|read|like this text, please help me generate a high quality example for this dataset.

For this task you will generate a good length answer using your best helpfulness and wisdom, and No need to include verbose or extraneous information.

Before I give you the text, I'll give you two examples:
Example 1:
### text:
What are the key considerations for scheduling and logistics when hosting a multi-show festival at a performing arts center like the Broward Center?
### persona:
A theater manager or events coordinator interested in understanding the operational aspects, facilities, and programming of performing arts centers, such as the Broward Center.
Example 2:
### text:
Compare and contrast the distribution of public services such as libraries, community centers, and public transportation in different neighborhoods of Halifax, and discuss how the municipal government's urban planning strategies impact access to these services for residents of varying socioeconomic backgrounds.
### persona:
An urban planner looking to understand the distribution and organization of public services in the Halifax Regional Municipality.

Now, it is your turn! Given the guidelines and examples above, please create a detailed and high quality description of the person who is most likely to say|listen|write|read|like|dislike this text:
### text:
{dialogue}

Your answer should be as follows:
### persona:
<a detailed description of the person>
'''
# description: <a detailed description of the person>
# reason: <reason for the person>
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

'''