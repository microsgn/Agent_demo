"""
第一版最小实现：
    实现 文本 ——> 向量
"""
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


def embed(texts:str|list[str],batch_size:int = 20)->list:
    out = []
    for i in range(0, len(texts), batch_size):
        chunks = texts[i:i+batch_size]
        response = dashscope.TextEmbedding.call(
            model = "qwen3.7-text-embedding",
            api_key = os.getenv("QWEN_API_KEY"),
            input = chunks,
            dimension = DIM,
        )
        resp = response.output
        ans = resp["embeddings"]
        for em in ans:
            em["text"] = chunks[em["text_index"]]
            out.append(em)
    return out

if __name__ == "__main__":
    input = ["如何处理好人际关系呢","华尧宽一定能成功"]
    response = embed(input)
    for resp in response:
        print(f"{resp["text_index"] + 1}:{resp['text']}")

