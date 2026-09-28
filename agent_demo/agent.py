"""
架构设计：
    Agent 的大脑部分写入 brain.py
    Agent 的工具部分写入 tool.py
    Agent 的记忆部分写入 memory.py

大脑应该包含什么？
    大脑应该主要是LLM，有与工具和记忆相接的接口
    工具主要就是定义的 工具API的接口，且要封装到LLM中（通过Agent_Loop实现），后续可以升级MCP

Agent_Loop:
    实现一个简易版的agent，核心思想在于先思考再行动，行动后观察结果是否能输出

"""

from typing import Dict,List,Any
import json,inspect
import httpx
from dotenv import load_dotenv
load_dotenv()
import os
from openai import OpenAI
from tool import get_weather

class Tool:
    """
    Tool类主要完成工具注册，还要完成工具的调用，并把返回结果返回给LLM
    """

    def __init__(self):
        self.tools = {}

    def register(self, func):
        self.tools[func.__name__] = func

    def execute(self, action:Dict):
        tool_name = action.get("action")
        args = action.get("args",{})
        return self.tools[tool_name](**args)
    def describe(self) -> str:
        """自动生成工具清单描述，给 LLM 看"""
        descs = []
        for name, func in self.tools.items():
            params = inspect.signature(func).parameters
            param_str = ", ".join(
                f"{p}: {v.annotation.__name__ if v.annotation is not inspect.Parameter.empty else 'any'}"
                for p, v in params.items()
            )
            doc = (func.__doc__ or "无描述").strip()
            descs.append(f"- {name}({param_str}): {doc}")
        return "\n".join(descs)

class SimpleAgent:
    def __init__(self,tool:Tool,max_iterations:int = 10,):
        self.max_iterations = max_iterations
        self.history : list[Dict] = []
        self.tool = tool
    def think(self,query:str):
        """LLM思考，决定下一步动作"""
        messages = [
            {"role":"system","content":f"""你是一个智能的Agent，可以使用的工具如下：
{self.tool.describe()}

你的任务：根据用户的请求，决定下一步动作。回复严格为 JSON 格式：
{{
    "action": "工具名，必须是上面列表中的工具名，或 'finish'",
    "args": {{"参数名": 参数值}},
    "thought": "你的思考过程"
    "answear": "你的最终回答"
}}
若任务已完成或无需工具即可回答，把 "action" 设为 "finish"且删除掉回答的args字段即可。
            """
            },
            {"role":"user","content":query}
        ]
        response = self.call_llm(messages)
        data = json.loads(response)
        self.history.append(data)
        return data

    def act(self,action:Dict)->Any:
        """执行动作"""
        resp = self.tool.execute(action)
        return json.loads(resp)
    def observe(self,tool_content:dict,query:str)->Any:
        """将结果反馈给 LLM"""
        messages = [
        {"role": "system", "content": f"""你是一个智能的Agent，可以使用的工具如下：
{self.tool.describe()}

你的任务：根据用户的请求，决定下一步动作。回复严格为 JSON 格式：
{{
    "action": "工具名，必须是上面列表中的工具名，或 'finish'",
    "args": {{"参数名": 参数值}},
    "thought": "你的思考过程"
    "answear": "你的最终回答"
}}
若任务已完成或无需工具即可回答，把 "action" 设为 "finish"且删除掉回答的args字段即可。    
        """
        },
        {"role":"user","content":query},
        {"role": "assistant", "content": json.dumps(self.history[-1], ensure_ascii=False)},  # 上一次决策，必须是字符串
        {"role": "user","content": f"工具返回结果：\n{json.dumps(tool_content, ensure_ascii=False)}\n请根据结果决定下一步，任务完成则 action 设为 finish。"},
    ]
        response = self.call_llm(messages)
        return json.loads(response)
    def run(self,user_input:str)->Any:
        """主循环"""
        decision = self.think(user_input)  # 先思考
        for i in range(self.max_iterations):
            print(f"执行第{i+1}次循环")
            if decision.get("action") == "finish":  # 先判断，再行动
                return decision  # 任务完成，直接返回
            result = self.act(decision)  # 有工具才执行
            decision = self.observe(result,user_input)  # 观察结果，再思考下一步
        return decision
    def call_llm(self,messages:List[Dict]):
        """调用LLM"""
        client = OpenAI(
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            base_url=os.getenv("DEEPSEEK_URL")
        )
        response = client.chat.completions.create(
            model="deepseek-v4-flash",
            messages=messages,
            stream=False,
            reasoning_effort="max",
            extra_body={"thinking": {"type": "disabled"}}
        )
        return response.choices[0].message.content

tool = Tool()
tool.register(get_weather)
agent = SimpleAgent(tool = tool)
resp = agent.run("帮我查询一下哈尔滨的天气")
print(resp)
