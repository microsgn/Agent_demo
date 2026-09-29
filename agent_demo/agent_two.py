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
from memory.short_term import ShortTermMemory
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
        self.mcp_clients = []
        self.mcp_tools = {}

    def bind_mcp(self, mcp: MCPClient, tools: dict) -> None:
        self.mcp_clients.append(mcp)

        for name, schema in tools.items():
            if name in self.tools or name in self.mcp_tools:
                raise ValueError(f"工具名称冲突：{name}")

            self.mcp_tools[name] = {
                "client": mcp,
                "schema": schema,
            }

    def register(self, func) -> None:
        name = func.__name__

        if name in self.tools or name in self.mcp_tools:
            raise ValueError(f"工具名称冲突：{name}")

        self.tools[name] = func

    def execute(self, action: Dict):
        tool_name = action.get("action")
        args = action.get("args", {})

        binding = self.mcp_tools.get(tool_name)
        if binding is not None:
            return binding["client"].call_tool(tool_name, args)

        local_tool = self.tools.get(tool_name)
        if local_tool is not None:
            return local_tool(**args)

        raise KeyError(f"未注册的工具：{tool_name}")

    def schemas(self) -> list[dict]:
        result = []

        # 本地 Python 工具
        for name, func in self.tools.items():
            signature = inspect.signature(func)
            param_docs = parse_param_docs(func.__doc__)
            properties = {}
            required = []

            for param_name, param in signature.parameters.items():
                param_type = mapping.get(param.annotation, "string")
                enum, description = param_docs.get(
                    param_name,
                    (None, None),
                )

                properties[param_name] = {
                    "type": param_type,
                    "description": (
                        description
                        or f"参数 {param_name}，类型为 {param_type}"
                    ),
                }

                if enum:
                    properties[param_name]["enum"] = enum

                if param.default is inspect.Parameter.empty:
                    required.append(param_name)

            result.append({
                "type": "function",
                "function": {
                    "name": name,
                    "description": (
                        func.__doc__ or "无描述"
                    ).strip().split("\n")[0],
                    "parameters": {
                        "type": "object",
                        "properties": properties,
                        "required": required,
                    },
                },
            })

        # MCP 工具：直接读取启动时缓存的 schema
        for name, binding in self.mcp_tools.items():
            tool_schema = binding["schema"]

            parameters = tool_schema.get("inputSchema") or {
                "type": "object",
                "properties": {},
            }

            result.append({
                "type": "function",
                "function": {
                    "name": name,
                    "description": tool_schema.get(
                        "description",
                        "无描述",
                    ),
                    "parameters": parameters,
                },
            })

        return result

class SimpleAgent:
    def __init__(self,tool:Tool,stm:ShortTermMemory,session_id:str,user_id:str = "default",max_iterations:int = 10,recent_limit:int = 20):
        self.max_iterations = max_iterations
        self.stm = stm              # shortTermMemory实例
        self.session_id = session_id
        self.user_id = user_id
        self.recent_limit = recent_limit
        self.tool = tool
    def run(self,user_input:str)->str:
        history = self.stm.recent(self.session_id,limit = self.recent_limit,user_id=self.user_id)
        messages:list[Dict] = [{"role":"system","content":"你是一个智能助手，需要实时信息的时候调用工具，并根据工具返回的真实结果回答用户"}]
        messages.extend(self.stm.to_llm_messages(history))
        messages.append({"role":"user","content":user_input})
        final = None
        for i in range(self.max_iterations):
            print(f"执行第{i+1}次循环")
            msg = self.call_llm(messages)
            if not msg.tool_calls:
                final = msg.content
                break
            else:
                messages.append(msg.model_dump(exclude_none = True))
                for tc in msg.tool_calls:
                    name = tc.function.name
                    args = json.loads(tc.function.arguments)
                    print(f"[调用]{name}{args}")
                    result = self.tool.execute({"action":name,"args":args})
                    if result is None:
                        result = "工具调用失败，请查询工具调用信息"
                    elif not isinstance(result,str):
                        result = json.dumps(result,ensure_ascii=False)
                    messages.append({
                        "role":"tool",
                        "tool_call_id":tc.id,
                        "content":result,
                    })
        if final is None:
            final =  f"达到最大循环次数仍未完成"

        self.stm.add(self.session_id,[{"role":"user","content":user_input},{"role":"assistant","content":final}],user_id=self.user_id)
        return final

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
    servers = load_mcp_server() or []
    tool = Tool()

    for _, client, discovered_tools in servers:
        tool.bind_mcp(client, discovered_tools)
    agent = SimpleAgent(tool = tool,stm = ShortTermMemory(),session_id = "001",user_id = "001")
    while True:
        user_input = input("用户:")
        if user_input == "exit":
            break
        resp = agent.run(user_input)
        print(resp)
