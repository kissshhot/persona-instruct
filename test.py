import os

os.environ["OPENAI_API_KEY"] = "sk-cBMDIQFHqCyqrGlSFb26F27dD9C4445d971a0673A73c7f9a" #输入网站发给你的转发key

os.environ["OPENAI_BASE_URL"] = "http://15.204.101.64:4000/v1"

from openai import OpenAI

client = OpenAI()

completion = client.chat.completions.create(

model="gpt-4",

messages=[

    {"role": "system", "content": "You are a helpful assistant."},

    {"role": "user", "content": "Hello!"}

]

)

print(completion)
print('test')