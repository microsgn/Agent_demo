import json
from mcp_client import MCPClient

def load_mcp_server(config_path = "mcp_config.json"):
    with open(config_path,encoding="utf-8") as f:
        cfg = json.load(f)
    servers = []
    for name,conf in cfg.get("mcpServers",{}).items():
        transport = str(conf.get("type","streamable_http")).replace("-","_")
        if transport not in ("streamable_http","http"):
            print(f"[跳过]{name}:不支持传输类型{conf.get('type')}")
            continue
        try:
            client = MCPClient(conf["url"],headers = conf.get("headers"),protocolVersion = conf.get("protocol_version", "2025-06-18"))
            client.initialize()
            tools = client.list_tools()
            servers.append((name,client,tools))
            print(f"[连接] {name}: {len(tools)} 个工具")
        except Exception as e:
            print(f"[失败] {name}: {e}")  # ← 一个服务器挂了不拖垮其它
    if servers:
        return servers
    else:
        return
if __name__ == "__main__":
    servers = load_mcp_server()
    print(servers)