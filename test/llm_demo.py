## 调用 DeepSeek 官方的SDK
from openai import OpenAI
import os
from dotenv import load_dotenv
load_dotenv()
client = OpenAI(
    api_key = os.getenv("DEEPSEEK_API_KEY"),
    base_url = os.getenv("DEEPSEEK_URL")
)
response = client.chat.completions.create(
    model = "deepseek-v4-flash",
    messages = [
        {"role":"system","content":"你是一个计算机和AI专家，解答我提出的疑惑"},
        {"role":"user","content":"你好呀"}
    ],
    stream = False,
    reasoning_effort="max",
    extra_body={"thinking":{"type":"disabled"}}
)

print(response)
print(response.choices[0].message.content)
