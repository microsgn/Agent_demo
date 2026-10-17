import httpx
import json
from typing import Any
import itertools

class MCPError(RuntimeError):
    pass
class MCPToolError(MCPError):
    pass

class MCPClient:
    def __init__(self,url,headers :dict | None = None,protocolVersion:str = "2025-06-18",read_timeout:float = 60.0):
        self.url = url
        self.session_id : str| None = None
        self.headers = dict(headers or {})
        self.protocolVersion = protocolVersion
        self.negotiated_protocol_version: str | None = None
        self._request_ids = itertools.count(1)
        self._tools_cache:dict[str,dict[str,Any]] | None = None
        timeout = httpx.Timeout(
            connect = 5.0,
            read = read_timeout,
            write = 10.0,
            pool = 5.0
        )
        self._http = httpx.Client(
            timeout = timeout,
            follow_redirects = True,
        )

    def close(self):
        self._http.close()

    def _next_id(self):
        return next(self._request_ids)

    def _make_headers(self)->dict[str,str]:
        headers = dict(self.headers)
        headers["Content-Type"] = "application/json"
        headers["Accept"] = "application/json,text/event-stream"

        if self.session_id:
            headers["Mcp-Session-Id"] = self.session_id

        if self.negotiated_protocol_version:
            headers["Mcp-Protocol-Version"] = (self.negotiated_protocol_version)

        return headers

    def _read_sse(self,response:httpx.Response,request_id:int)->dict[str,Any]:
        data_lines :list[str] = []
        for line in response.iter_lines():
            if line.startswith("data:"):
                data_lines.append(line[5:].lstrip())
                continue

            if line != "" or not data_lines:
                continue

            message = json.loads("\n".join(data_lines))
            data_lines.clear()

            if message.get("id") == request_id:
                return message

        raise MCPError(
            f"SSE 流结束，但没有收到请求 {request_id} 的响应"
        )


    def _post(
            self,
            payload: dict[str, Any],
    ) -> dict[str, Any] | None:
        method = payload.get("method", "<unknown>")

        try:
            with self._http.stream(
                    "POST",
                    self.url,
                    headers=self._make_headers(),
                    json=payload,
            ) as response:
                session_id = response.headers.get("Mcp-Session-Id")
                if session_id:
                    self.session_id = session_id

                if response.is_error:
                    response.read()
                    raise MCPError(
                        f"{method} 返回 HTTP {response.status_code}: "
                        f"{response.text[:300]}"
                    )

                # JSON-RPC notification 没有 id，服务端不应该返回结果。
                if "id" not in payload:
                    response.read()
                    return None

                content_type = response.headers.get(
                    "Content-Type", ""
                ).lower()

                if content_type.startswith("application/json"):
                    response.read()
                    message = response.json()
                elif content_type.startswith("text/event-stream"):
                    message = self._read_sse(
                        response,
                        payload["id"],
                    )
                else:
                    response.read()
                    raise MCPError(
                        f"{method} 返回了不支持的 Content-Type: "
                        f"{content_type!r}"
                    )

        except httpx.TimeoutException as exc:
            raise MCPError(
                f"{method} 调用超时：{type(exc).__name__}"
            ) from exc
        except httpx.HTTPError as exc:
            raise MCPError(
                f"{method} 网络请求失败：{type(exc).__name__}"
            ) from exc

        error = message.get("error")
        if error is not None:
            raise MCPError(
                f"{method} JSON-RPC 错误："
                f"{json.dumps(error, ensure_ascii=False)}"
            )

        if "result" not in message:
            raise MCPError(f"{method} 响应缺少 result 字段")

        return message


    def initialize(self) -> dict[str, Any]:
        response = self._post({
            "jsonrpc": "2.0",
            "id": self._next_id(),
            "method": "initialize",
            "params": {
                "protocolVersion": self.protocolVersion,
                "capabilities": {},
                "clientInfo": {
                    "name": "agent-two",
                    "version": "1.0",
                },
            },
        })

        result = response["result"]
        negotiated = result.get("protocolVersion")
        if not negotiated:
            raise MCPError("initialize 响应缺少 protocolVersion")

        self.negotiated_protocol_version = negotiated

        self._post({
            "jsonrpc": "2.0",
            "method": "notifications/initialized",
        })

        return result


    def list_tools(
            self,
            refresh: bool = False,
    ) -> dict[str, dict[str, Any]]:
        if self._tools_cache is not None and not refresh:
            return dict(self._tools_cache)

        tools: dict[str, dict[str, Any]] = {}
        cursor: str | None = None
        seen_cursors: set[str] = set()

        while True:
            params = {}
            if cursor is not None:
                params["cursor"] = cursor

            response = self._post({
                "jsonrpc": "2.0",
                "id": self._next_id(),
                "method": "tools/list",
                "params": params,
            })

            result = response["result"]

            for tool in result.get("tools", []):
                name = tool.get("name")
                if not name:
                    raise MCPError("tools/list 返回了没有名称的工具")
                if name in tools:
                    raise MCPError(f"发现重名 MCP 工具：{name}")
                tools[name] = tool

            cursor = result.get("nextCursor")
            if not cursor:
                break

            if cursor in seen_cursors:
                raise MCPError("tools/list 返回了重复游标")
            seen_cursors.add(cursor)

        self._tools_cache = tools
        return dict(tools)


    def call_tool(self, name: str, arguments: dict) -> str:
        response = self._post({
            "jsonrpc": "2.0",
            "id": self._next_id(),
            "method": "tools/call",
            "params": {
                "name": name,
                "arguments": arguments,
            },
        })

        result = response["result"]
        parts: list[str] = []

        for block in result.get("content", []):
            if block.get("type") == "text":
                parts.append(str(block.get("text", "")))
            else:
                parts.append(
                    json.dumps(block, ensure_ascii=False)
                )

        structured = result.get("structuredContent")
        if structured is not None:
            parts.append(
                json.dumps(structured, ensure_ascii=False)
            )

        text = "\n".join(part for part in parts if part)

        if result.get("isError"):
            raise MCPToolError(
                f"工具 {name} 执行失败：{text or '未提供错误信息'}"
            )

        return text


    def schema(self) -> list[dict[str, Any]]:
        schemas = []

        # 这里命中缓存，不再发远程 tools/list。
        for tool in self.list_tools().values():
            schemas.append({
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool.get(
                        "description", "无描述"
                    ),
                    "parameters": tool.get(
                        "inputSchema",
                        {
                            "type": "object",
                            "properties": {},
                        },
                    ),
                },
            })

        return schemas
