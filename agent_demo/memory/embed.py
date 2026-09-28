"""
嵌入函数流程：
    1. 先写一个sqlite进行存储，如果数据库里面存储的有便不需要每次都调用模型了，一是成本更低，二是速度更快
    2. 
"""
import dashscope
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

def _cache_init():
    global _initialized
    if _initialized:
        return
    with closing(sqlite3.connect(_CACHE_DB, timeout=30)) as conn, conn:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("""CREATE TABLE IF NOT EXISTS cache (
            hash TEXT PRIMARY KEY, text TEXT, vec BLOB, created_at REAL)""")
    _initialized = True


def key(text: str) -> str:
    return hashlib.sha256(f"{EMBED_MODEL}|{DIM}|{text}".encode()).hexdigest()


def fetch(text):
    texts = [text] if isinstance(text, str) else list(text)
    texts = [t for t in texts if t and t.strip()]     # 过滤空串
    if not texts:
        return []

    _cache_init()
    out = [None] * len(texts)
    missed_index = []
    with closing(sqlite3.connect(_CACHE_DB, timeout=30)) as conn:
        for i, t in enumerate(texts):
            row = conn.execute("SELECT vec FROM cache WHERE hash=?", (key(t),)).fetchone()
            if row:
                v = _unpack(row[0])
                if len(v) == DIM:
                    out[i] = v
                else:
                    missed_index.append(i)            # 维度不符 → 重算
            else:
                missed_index.append(i)

    missing = [texts[i] for i in missed_index]
    for start in range(0, len(missing), BATCH_SIZE):
        chunk = missing[start:start + BATCH_SIZE]
        chunk_idx = missed_index[start:start + BATCH_SIZE]

        resp = _call_with_retry(chunk)
        ordered = sorted(resp["output"]["embeddings"], key=lambda x: x["text_index"])
        if len(ordered) != len(chunk):
            raise RuntimeError(f"返回条数不匹配: 请求{len(chunk)} 返回{len(ordered)}")

        for k, item in enumerate(chunk_idx):
            v = ordered[k]["embedding"]
            if len(v) != DIM:
                raise RuntimeError(f"维度错误: {len(v)} != {DIM}")
            out[item] = v

    with closing(sqlite3.connect(_CACHE_DB, timeout=30)) as conn, conn:
        conn.executemany(
            "INSERT OR REPLACE INTO cache(hash,text,vec,created_at) VALUES(?,?,?,?)",
            [(key(texts[i]), texts[i], _pack(out[i]), time.time()) for i in missed_index],
        )
    return out


def _call_with_retry(chunk, retries=3):
    for attempt in range(retries):
        resp = dashscope.TextEmbedding.call(
            model=EMBED_MODEL, api_key=os.getenv("QWEN_API_KEY"),
            input=chunk, dimension=DIM,
        )
        if resp.status_code == HTTPStatus.OK:
            return resp
        if attempt == retries - 1:
            msg = getattr(resp, "message", None) or resp.get("message", "unknown")
            raise RuntimeError(f"请求失败: status={resp.status_code} msg={msg}")
        time.sleep(0.5 * 2 ** attempt)