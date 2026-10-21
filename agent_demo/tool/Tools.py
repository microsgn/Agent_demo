from MCP.mcp_client import MCPClient
from typing import Dict
import inspect
import re
from .permission import PermissionGate
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

    def __init__(self,gate:PermissionGate):
        self.tools = {}
        self.mcp_clients = []
        self.mcp_tools = {}
        self.gate = gate

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
        deny = self.gate.guard(tool_name, args)
        if deny is not None:
            return deny

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