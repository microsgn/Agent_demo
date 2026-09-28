from typing import Dict,List,Any
import json,inspect
from dotenv import load_dotenv
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # 让 memory 包可被找到
from memory.Memory import Memory
from agent_demo.MCP import MCP
from mcp_loader import load_mcp_server
load_dotenv()
import os
import re
from openai import OpenAI
from tool import get_weather
from mcp_client import MCPClient
mapping = {
    str:"string",
    int:"integer",
    float:"number",
    bool:"boolean",
}
PARAM_RE = re.compile(r"^\s*(\w+)\s*:\s*([^\n]+)$", re.MULTILINE)

def parse_param_docs(doc:str)->dict:
    """从 docstring 解析参数说明，返回{参数名：(enum列表，说明)}"""
    out = {}
    for name, body in PARAM_RE.findall(doc or ""):
        body = body.strip()
        if " - " in body:
            head, desc = body.split(" - ", 1)
        else:
            head, desc = body, ""
        parts = [p.strip() for p in head.split("/") if p.strip()]
        enum = parts if len(parts) > 1 else None  # 只有一个值就不当枚举
        out[name] = (enum, desc or head)
    return out

class Tool:
    """
    Tool类主要完成工具注册，还要完成工具的调用，并把返回结果返回给LLM
    """

    def __init__(self):
        self.tools = {}
        self.mcp_tool = []
        self.mcp = []
    def binds_mcp(self,mcp:MCPClient):
        self.mcp.append(mcp)
    def register(self, func):
        self.tools[func.__name__] = func
    def register_mcp_tool(self):
        for mcp in self.mcp:
            self.mcp_tool.append(mcp.list_tools())
    def execute(self, action:Dict):
        tool_name = action.get("action")
        args = action.get("args",{})
        for mcp in self.mcp:
            if tool_name in mcp.list_tools():
                return mcp.call_tool(tool_name,args)
        return self.tools[tool_name](**args)
    def schemas(self)->list:
        result = []
        for name,func in self.tools.items():
            sig = inspect.signature(func)
            param_docs = parse_param_docs(func.__doc__)
            props,required = {},[]
            for pname,p in sig.parameters.items():
                t = mapping.get(p.annotation,"string")
                enum, desc = param_docs.get(pname, (None, None))
                props[pname] = {
                    "type":t,
                    "description":f"参数{pname},类型{t}"
                }
                if enum:
                    props[pname]["enum"] = enum
                if p.default is inspect.Parameter.empty:
                        required.append(pname)
            result.append({
                "type":"function",
                "function":{
                    "name":name,
                    "description":(func.__doc__ or "无描述").strip().split("\n")[0],
                    "parameters":{
                        "type":"object",
                        "properties":props,
                        "required":required,
                    },
                },
            })
        ### 第二部分：将MCP工具的响应转换为Openai的function calling
        ans = result
        for mcp in self.mcp:
            ans = ans + mcp.schema()
        return ans

class SimpleAgent:
    def __init__(self,tool:Tool,max_iterations:int = 10,):
        self.max_iterations = max_iterations
        self.history : list[Dict] = []
        self.tool = tool
    def run(self,user_input:str):
        self.history = [
            {"role":"system","content":"你是一个智能助手，需要实时信息的时候调用工具，并根据工具返回的真实结果回答用户"},
            {"role":"user","content":user_input},
        ]
        for i in range(self.max_iterations):
            print(f"执行第{i+1}次循环")
            msg = self.call_llm(self.history)
            if not msg.tool_calls:
                return msg.content
            else:
                self.history.append(msg.model_dump(exclude_none = True))
                for tc in msg.tool_calls:
                    name = tc.function.name
                    args = json.loads(tc.function.arguments)
                    print(f"[调用]{name}{args}")
                    result = self.tool.execute({"action":name,"args":args})
                    if result is None:
                        result = "工具调用失败，请查询工具调用信息"
                    elif not isinstance(result,str):
                        result = json.dumps(result,ensure_ascii=False)
                    self.history.append({
                        "role":"tool",
                        "tool_call_id":tc.id,
                        "content":result,
                    })
        return f"达到最大循环次数仍未完成"
    def call_llm(self,messages:List[Dict]):
        """调用LLM"""
        client = OpenAI(
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            base_url=os.getenv("DEEPSEEK_URL")
        )
        response = client.chat.completions.create(
            model="deepseek-v4-flash",
            messages=messages,
            tools = self.tool.schemas(),
            tool_choice="auto",
            stream=False,
            reasoning_effort="max",
            extra_body={"thinking": {"type": "disabled"}}
        )
        return response.choices[0].message

if __name__ == "__main__":
    servers = load_mcp_server()
    clients = [c for _, c, _ in servers]
    tool = Tool()
    for mcp in clients:
        tool.binds_mcp(mcp)
    tool.register_mcp_tool()
    memory = Memory()
    tool.register(memory.recall)
    tool.register(memory.remember)
    agent = SimpleAgent(tool = tool)
    resp = agent.run("我是谁？")
    print(resp)
