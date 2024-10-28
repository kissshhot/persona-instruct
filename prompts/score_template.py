score_template = '''
We would like you to evaluate and rate the difficulty and complexity of the following question. You
should give an overall score on a scale of 1 to 10, where a higher score indicates higher difficulty and
complexity. You must just give a score without any other reasons.
### Question:
{instruct}
Your answer should be as follows:
### Score:
<The score you gave>
### Reason:
<The reason you gave this score>
'''

choose_template='''
Here's the instruct:
### Instruct:
{instruct}
Please choose the most correct and clear answer from the three responses below:
### response A:
{responseA}
### response B:
{responseB}
### response C:
{responseC}
Your answer should be as follows:
## Choice:
< A|B|C >
'''