"""
第一版最小实现：
    实现 文本 ——> 向量
"""
from urllib.error import HTTPError

import random
import dashscope
import json
from http import HTTPStatus
import os
import hashlib
import time
import struct
from dotenv import load_dotenv
load_dotenv()
dashscope.base_http_api_url = os.getenv("QWEN_BASE_URL")
DIM = 1024
EMBED_MODEL = "qwen3.7-text-embedding"
BATCH_SIZE = 20
_CACHE_DB = "embed_cache.db"
_initialized = False
max_retries = 3
RETRYABLE_STATUS = {429, 500, 502, 503, 504}

def _pack(vec:list[float])->bytes:
    return struct.pack(f"<{len(vec)}f", *vec)
def _unpack(blob:bytes)->list[float]:
    if len(blob) % 4 != 0:
        raise ValueError(f"缓存数据损坏:{len(blob)}字节")
    return list(struct.unpack(f"<{len(blob)//4}f", blob))
import sqlite3
from contextlib import closing


def fetch(text):
    pass


def embed(texts:list[str],batch_size:int = 20)->list:
    out = []
    for i in range(0, len(texts), batch_size):
        chunks = texts[i:i+batch_size]
        resp = call_llm(chunks)
        ans = resp["embeddings"]
        for em in ans:
            em["text"] = chunks[em["text_index"]]
            em["id"] = em["text_index"] + i
            out.append(em)
    output = sorted(out, key = lambda x : x["id"])
    return output

## 处理网络请求出错修改
def call_llm(texts:list[str])->dict:
    status = None
    for attempt in range(max_retries):
        try:
            response = dashscope.TextEmbedding.call(
                model = "qwen3.7-text-embedding",
                api_key = os.getenv("QWEN_API_KEY"),
                input = texts,
                dimension = DIM,
            )
            status = response.status_code
            if status == HTTPStatus.OK:
                return response.output
            elif status is not None and status not in RETRYABLE_STATUS:
                raise RuntimeError(
                    f"embedding 请求失败(不可重试): status={status} "
                    f"code={getattr(response, 'code', None)} "
                    f"message={getattr(response, 'message', None)}"
                )
        except Exception as e:
            print(e)
        if attempt < max_retries - 1:
    # 指数退避 + 随机抖动（抖动防止所有客户端同时重试形成尖峰）
            time.sleep(0.5 * 2 ** attempt + random.uniform(0, 0.3))
    raise RuntimeError(f"embedding 重试 {max_retries} 次仍失败")

if __name__ == "__main__":
    input = ["如何处理好人际关系呢","华尧宽一定能成功","你好"]
    response = embed(input)
    for resp in response:
        print(f"{resp["text_index"] + 1}:{resp['text']}")

