from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()
import os

client = OpenAI(
    api_key = os.getenv("DEEPSEEK_API_KEY"),
    base_url = os.getenv("DEEPSEEK_URL")
)
class llm:
    def __init__(self,model):
        self.model = model

    def think(self,query):
        try:
            response = client.chat.completions.create(
                model = self.model,
                messages = [
                    {"role":"user","content":f"{query}"}
                ],
                temperature = 0.7,
                max_tokens = 1000,
                stream = False
            )
            reasoning = response.choices[0].message.content
            return reasoning.strip()
        except Exception as e:
            return f"思考过程出错:{e}"

### 简单测试一下
if __name__ == "__main__":
    brain = llm("deepseek-v4-flash")
    query = "你好，介绍一下自己"
    print("测试提问：",query)
    print("大脑回复：",brain.think(query))
