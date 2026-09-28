import httpx
class MCPClient:
    def __init__(self,url,headers = None,protocolVersion = "2025-06-18"):
        self.url = url
        self.session_id = None
        self.headers = headers or {}
        self.protocolVersion = protocolVersion

    def _post(self,payload):
        headers = {
            "Content-Type": "application/json",
            "Accept":"application/json,text/event-stream",
        }
        headers.update(self.headers)
        if self.session_id:
            headers["Mcp-Session-Id"] = self.session_id
        with httpx.Client() as c:
            r = c.post(self.url,headers = headers,json = payload)
        if "Mcp-Session-Id" in r.headers:
            self.session_id = r.headers["Mcp-Session-Id"]
        if "id" not in payload:
            return None
        return r.json()

    def initialize(self):
        self._post({
            "jsonrpc":"2.0",
            "id":1,
            "method":"initialize",
            "params":{
                "protocolVersion":self.protocolVersion,
                "capabilities":{},
                "clientInfo":{"name":"agent-two","version":"1.0"},
            },
        })
        self._post({"jsonrpc":"2.0","method":"notification/initialized"})

    def list_tools(self):
        resp = self._post({
            "jsonrpc":"2.0",
            "id":2,
            "method":"tools/list",
        })
        return {t["name"]: t for t in resp["result"]["tools"]}

    def call_tool(self,name,arguments):
        resp = self._post({
            "jsonrpc":"2.0",
            "id":3,
            "method":"tools/call",
            "params":{
                "name":name,
                "arguments":arguments,
            }
        })
        return resp["result"]["content"][0]["text"]

    def schema(self):
        resp = self._post({
            "jsonrpc":"2.0",
            "id":4,
            "method":"tools/list",
        })
        t = resp["result"]["tools"]
        result = []
        for tool in t:
            param = tool["inputSchema"]
            schema = {
                "type":"function",
                "function":{
                    "name":tool["name"],
                    "description":tool["description"],
                    "parameters":{
                        "type":"object",
                        "properties":param["properties"],
                        "required":param["required"]
                    }
                }
            }
            result.append(schema)
        return result
