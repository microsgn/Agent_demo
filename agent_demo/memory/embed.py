"""
第一版最小实现：
    实现 文本 ——> 向量
"""
from urllib.error import HTTPError
import requests
import random
import dashscope
import json

from _pytest.assertion import truncate

from models.embed_item import  EmbeddingItem
from http import HTTPStatus
import os
import hashlib
import time
import struct
from dotenv import load_dotenv
load_dotenv()
dashscope.base_http_api_url = os.getenv("QWEN_BASE_URL")
DIM = 1024
EMBED_MODEL = os.getenv("OLLMA_MODEL")
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

class EmbeddingError(RuntimeError):
    """embedding 相关错误（调用方可以精确 catch 这个）"""

def fetch(text):
    pass

def _cache_init()->None:
    with closing(sqlite3.connect(_CACHE_DB,timeout = 30)) as conn,conn:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("""
            CREATE TABLE IF NOT EXISTS cache (
                hash TEXT PRIMARY KEY,
                text TEXT NOT NULL,
                vec  BLOB NOT NULL,
                created_at REAL NOT NULL
            )
        """)

def _key(text:str)->str:
    """缓存键 = 模型 + 维度 + 原文。换模型/换维度会自动失效."""
    return hashlib.sha256(f"{EMBED_MODEL}|{DIM}|{text}".encode()).hexdigest()

def _cache_get(keys:list[str])->dict[str,list[float]]:
    out:dict[str,list[float]] = {}
    for start in range(0,len(keys),500):
        chunks = keys[start:start+500]
        if not chunks:
            continue
        ph = ",".join(["?"]*len(chunks))
        with closing(sqlite3.connect(_CACHE_DB,timeout = 30)) as conn:
            rows = conn.execute(
                f"SELECT hash,vec FROM cache WHERE hash IN ({ph})",
                chunks
            ).fetchall()
        for h,blob in rows:
            try:
                vec = _unpack(blob)
            except ValueError:
                continue
            if len(vec) == DIM:
                out[h] = vec
    return out

def _cache_put(items:dict[str,tuple[str,list[float]]])->None:
    """批量写"""
    if not items:
        return
    rows = [(h,t,_pack(v),time.time())for h,(t,v) in items.items()]
    with closing(sqlite3.connect(_CACHE_DB,timeout = 30)) as conn,conn:
        conn.executemany(
            """INSERT OR REPLACE INTO cache(hash,text,vec,created_at) VALUES(?,?,?,?)""",
            rows,
        )

def embed(texts:list[str],batch_size:int = 20)->list:
    """返回texts对应的向量"""
    if len(texts) == 0:
        return []
    _cache_init()
    keys = [_key(t) for t in texts]
    # 1. 查缓存
    got = _cache_get(keys)
    # 2. 存储未命中的下标，哈希法可以去重，去掉重复存储的
    miss_pos :dict[str,int] = {}
    for i,k in enumerate(keys):
        if k not in got:
            miss_pos[k] = i
    miss_hash = list(miss_pos)
    # 3. 处理未命中的
    for i in range(0,len(miss_hash),batch_size):
        batch = miss_hash[i:i+batch_size]
        chunks = [texts[miss_pos[k]] for k in batch]
        resp = call_llm(chunks)
        seen : set[int] = set()
        for em in resp:
            local = em.text_index
            if not isinstance(local, int) or not 0 <= local < len(chunks):
                raise EmbeddingError(f"text_index越界：{local!r}")
            if local in seen:
                raise EmbeddingError(f"重复存入")
            seen.add(local)

            vec = em.embedding
            if len(vec) != DIM:
                raise EmbeddingError(f"向量维度不对")
            got[batch[local]] = vec
        if len(seen) != len(chunks):
            raise EmbeddingError(f"响应数量不匹配")
        _cache_put({k:(texts[miss_pos[k]],got[k])for k in batch})
    return [got[k] for k in keys]

## 处理网络请求出错修改
def call_llm(texts:list[str])->list[EmbeddingItem]:
    status = None
    for attempt in range(max_retries):
        try:
            response_list = []
            response = requests.post(
                url = os.getenv("OLLMA_URL"),
                json ={
                    "model":os.getenv("OLLMA_MODEL"),
                    "input":texts,
                    "truncate":False,
                },
                timeout = 30,
            )
            status = response.status_code
            if status == HTTPStatus.OK:
                data = response.json()["embeddings"]
                return [
                    EmbeddingItem(text_index = i,embedding = v)
                    for i ,v in enumerate(data)
                ]
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


