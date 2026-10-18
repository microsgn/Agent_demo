import json

import tiktoken

WINDOW = 1_000_000
MAX_TOKENS = 65_536
SAFETY_TOKENS = 8_192
COMPACTION_RATIO = 0.2
ENCODE = tiktoken.get_encoding("cl100k_base")
def estimate_tokens(context:list[dict],tools:list[dict]|None = None)->int:
    """对整份请求进行近似估计，不逐个字段拆开计算"""
    payload = {
        "messages":context,
        "tools":tools or []
    }
    text = json.dumps(payload,ensure_ascii=False,separators=(",", ":"),allow_nan=False)
    token_ids = ENCODE.encode_ordinary(text)
    return len(token_ids)

def remain_tokens(input_tokens:int,max_tokens:int = MAX_TOKENS,windows:int = WINDOW)->int:
    need = SAFETY_TOKENS + max_tokens + input_tokens
    remain_token = windows - need
    return remain_token

def need_compaction(remain_tokens:int,windows:int = WINDOW)->bool:
    if remain_tokens < windows * 0.2:
        return True
    else:
        return False