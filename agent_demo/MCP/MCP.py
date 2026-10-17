"""
实现一个手写的 MCP 服务器
    主要核心：
        * 获取工具的JSON定义
        * 处理JSON-RPC格式的请求
            ** initialize
            ** tools/list
            ** tools/call
"""
import threading
import uuid
from queue import Queue
import json,inspect
import traceback
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from tool import get_weather

supported_protocol_versions = ["2025-06-18"]
mapping = {
    str:"string",
    int:"integer",
    float:"number",
    bool:"boolean",
}

def add(a:int,b:int)->int:
    return a+b

## 想要实现使用MCP服务器添加工具的话，就不能提前定义一个工具来使用
class MCP:
    def __init__(self,host,port):
        self.tools = {}
        self.host = host
        self.port = port
        self.lock = threading.Lock()
        self.session : dict[str,Queue] = {}

    def tool(self,func):
        self.tools[func.__name__] = func
        return func

    def get_tool_json(self)->list:
        """获取mcp服务器里边工具的 JSON Schema"""
        result = []
        for name,func in self.tools.items():
            signature = inspect.signature(func)
            props,required = {},[]
            for pname,param in signature.parameters.items():
                t = mapping.get(param.annotation,"string")
                props[pname] = {
                    "type": t,
                    "description":f"参数{pname},类型为{t}",
                }
                if param.default is inspect.Parameter.empty:
                    required.append(pname)
            result.append({
                "name":name,
                "description":(func.__doc__ or "无描述").strip().split("\n")[0],
                "inputSchema":{
                    "type":"object",
                    "properties":props,
                    "required":required,
                },
            })
        return result
    def push(self,session_id:str,message:dict)->bool:
        with self.lock:
            queue = self.session.get(session_id)
        if queue is None:
            return False
        queue.put(message)
        return True

    def handler_json_rpc(self,request:dict,session_id:str|None)->dict:
        """=== 处理json-rpc 请求===，牢记三大核心"""
        request_id = request.get("id")
        method = request.get("method")
        params = request.get("params",{})
        ### 方法1: initialize —— 初始化握手，客户端启动时候调用
        if method == "initialize":
            request_version = params.get("protocolVersion")
            if request_version in supported_protocol_versions:
                return {
                    "jsonrpc":"2.0",
                    "id":request_id,
                    "result":{
                        "protocolVersion":request_version,
                        "capabilities":{"tools":{}},
                        "serverInfo":{"name":"My-Custom-MCP-Server","version":"1.0.0"}
                    }
                }
            else:
                return {
                    "jsonrpc":"2.0",
                    "id":request_id,
                    "error":{
                        "code":-32602,
                        "message":"Unsupported protocol version",
                        "data":{
                            "supported":supported_protocol_versions,
                            "requested":request_version
                        }
                    }
                }
        elif method =="notifications/initialized":
            print("初始化完成")
        elif method == "ping":
            return {
                "jsonrpc":"2.0",
                "id":request_id,
                "result":"pong"
            }
        elif method == "tools/list":
            try:
                tool_list = self.get_tool_json()
                return {
                    "jsonrpc":"2.0",
                    "id":request_id,
                    "result":{
                        "tools":tool_list,
                    }
                }
            except Exception as e:
                return {
                    "jsonrpc":"2.0",
                    "id":request_id,
                    "error":{
                        "code":-32603,
                        "message":f"展示工具失败：{str(e)}"
                    }
                }
        elif method == "tools/call":
            token = (params.get("_meta")or{}).get("progressToken")
            if token:
                self.push(session_id,{
                    "jsonrpc":"2.0",
                    "method":"notifications/progress",
                    "params":{
                        "progressToken":token,
                        "progress":50,
                        "total":100
                    }
                })
            try:
                tool_name = params.get("name")
                arguments = params.get("arguments",{})
                # 校验一下工具是否存在
                if tool_name not in self.tools:
                    return {
                        "jsonrpc":"2.0",
                        "id":request_id,
                        "error":{
                            "code":-32601,
                            "message":f"工具{tool_name}不存在",
                        }
                    }
                func = self.tools[tool_name]
                result = func(**arguments)

                return {
                    "jsonrpc":"2.0",
                    "id":request_id,
                    "result":{
                        "content":[
                            {
                                "type":"text",
                                "text":str(result),
                            }
                        ]
                    }
                }
            except Exception as e:
                return {
                    "jsonrpc":"2.0",
                    "id":request_id,
                    "error":{
                        "code":-32500,
                        "message":f"工具执行失败{str(e)}",
                        "data":traceback.format_exc()
                    }
                }
        else:
            return {
                "jsonrpc":"2.0",
                "id":request_id,
                "error":{
                    "code":-32601,
                    "message":f"Method{method} not supported",
                }
            }
def make_handler(mcp:MCP):
    class MCPHandler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"   # 必填：默认 HTTP/1.0，SSE 连接无法正确保活
        def __init__(self,*args,**kwargs):
            self.mcp = mcp
            super().__init__(*args,**kwargs)

        def do_POST(self):
            if self.path !="/mcp":
                self.send_json(404,{"error":"not found"})
                return
            n = int(self.headers.get("content-length",0))
            try:
                request = json.loads(self.rfile.read(n))
            except json.JSONDecodeError:
                self.send_json(400,{"jsonrpc": "2.0", "id": None,
                                     "error": {"code": -32700, "message": "Parse error"}})
                return
            session_id = self.headers.get("Mcp-Session-Id")
            if "id" not in request:
                if session_id and session_id not in self.mcp.session:
                    self.send_json(404, {"error": "invalid session"})
                    return
                self.send_response(202)
                self.send_header("Content-Length", "0")
                self.end_headers()
                return

            if request.get("method") == "initialize":
                if session_id not in self.mcp.session:
                    session_id = uuid.uuid4().hex
                with self.mcp.lock:
                    self.mcp.session.setdefault(session_id,Queue())
            else:
                if not session_id or session_id not in self.mcp.session:
                    self.send_json(404, {"jsonrpc": "2.0", "id": request.get("id"),
                                         "error": {"code": -32001, "message": "invalid session"}})
                    return

            resp = self.mcp.handler_json_rpc(request,session_id)
            self.send_json(200,resp,session_id=session_id)

        def do_GET(self):
            if self.path != "/mcp":
                self.send_json(404,{"error":"not found"})
                return
            self.handler_sse()

        def handler_sse(self):
            # 先获取请求的Id
            session_id = self.headers.get("Mcp-Session-Id")
            # 如果是新的请求Id,便给其赋值一个Id
            if not session_id or session_id not in self.mcp.session:
                # 兼容"先 GET 再 initialize"的客户端：新建会话并回传
                session_id = uuid.uuid4().hex
                with self.mcp.lock:
                    self.mcp.session[session_id] = Queue()
            queue = self.mcp.session[session_id]
            ## SSE 完成的任务主要就是推送
            self.send_response(200)
            self.send_header("Content-type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "keep-alive")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("X-Accel-Buffering", "no")
            self.send_header("Mcp-Session-Id", session_id)
            self.end_headers()
            try:
                while True:
                    message = queue.get(timeout = 15)
                    frame = f"event: message\ndata: {json.dumps(message, ensure_ascii=False)}\n\n"
                    self.wfile.write(frame.encode("utf-8"))
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass
            finally:
                with self.mcp.lock:
                    self.mcp.session.pop(session_id)

        def send_json(self,status:int,data:dict,session_id:str = None):
            body = json.dumps(data,ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            if session_id:
                self.send_header("Mcp-Session-Id",session_id)
            self.send_header("Content-Type","application/json")
            self.send_header("Content-Length",str(len(body)))
            self.end_headers()
            self.wfile.write(body)
    return MCPHandler

def main():
    mcp = MCP(host = "127.0.0.1", port = 8080)
    mcp.tool(add)   # 注册工具：等价于把 @mcp.tool 写在 add 定义上方；多个工具就多写几行
    mcp.tool(get_weather)
    http_server = ThreadingHTTPServer((mcp.host,mcp.port),make_handler(mcp))
    print(f"MCP 服务器已启动: http://{mcp.host}:{mcp.port}/mcp")
    try:
        http_server.serve_forever()
    except KeyboardInterrupt:
        print("\n服务器终止")
    finally:
        http_server.server_close()

if __name__ == "__main__":
    main()