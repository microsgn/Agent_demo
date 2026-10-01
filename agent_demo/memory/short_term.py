import redis
import os
import json
import numpy as np
from . import embed
from .bm25 import BM25
from .tokenizer import tokenize
import time
from dotenv import load_dotenv
load_dotenv()

HOST = os.getenv("REDIS_HOST")
PORT = os.getenv("REDIS_PORT")
key_prefix = "mem"
max_turns = 1000
max_chars = 2000
tool_chars = 600
expire_time = 7 * 24 * 3600
store_roles = {"user","assistant","tool"}
SCAN = 1000
BM25_TOP = 50
ALPHA = 0.4
MIN_VEC_SIM = 0.50      # 最低余弦相似度，低于此值视为"都不相关"。与 Memory.MIN_BEST_SIM 同尺

class ShortTermMemory():
    def __init__(self,host:str = HOST,port:str = PORT,prefix:str = key_prefix):
        self.store = redis.Redis(host = host,port = int(port),decode_responses = True)
        self._raw = redis.Redis(host=host, port=int(port), decode_responses=False)
        self.prefix = prefix

    def key(self,session_id:str,user_id:str = "default")->tuple[str,str]:
        base = f"{key_prefix}:{session_id}:{user_id}"
        return f"{base}:turns",f"{base}:vec"

    def add_filter(self, message: list[dict],prev:dict|None = None) -> list[dict]:
        """返回规范化后的消息；返回 None = 这条不该入库。
        只负责三件事，不做任何语义判断：
          1. 形状：role 白名单 + 非空 content + 字段规整
          2. 体积：普通 2000 / 工具结果 600 字符截断
          3. 去重：与上一条完全相同（同 role 同 content）时丢弃
        """
        resp = []
        last = prev
        for msg in message:
            if not isinstance(msg,dict):
                continue
            role = str(msg.get("role") or "" ).strip().lower()
            if role not in store_roles:
                continue
            content = msg.get("content") or ""
            if not isinstance(content,str) or not content.strip():
                continue

            limit = tool_chars if role == "tool" else max_chars
            ans = {
                "mid":self._mid(),
                "role":role,
                "content":content[:limit],
                "ts":float(msg.get("ts") or time.time()),
            }
            if role == "tool" and msg.get("name"):
                ans["name"] = str(msg.get("name"))
            if content[:limit] != content:
                ans["truncated"] = True

            if last and last.get("role") == role and last.get("content") == content:
                continue
            resp.append(ans)
            last = ans
        return resp
    _last_ms = 0
    @classmethod
    def _mid(cls)->str:
        """毫秒时间戳；同一毫秒内多次调用则递增，保证 field 唯一"""
        ms = int(time.time()*1000)
        if ms <= cls._last_ms:
            ms = cls._last_ms + 1
        cls._last_ms = ms
        return str(ms)


    def add(self, session_id:str,message: list[dict],prev:dict|None = None,user_id:str = "default")->bool:
        """"""
        item = self.add_filter(message,prev)
        if not item:
            return False
        k_list,k_vec = self.key(session_id,user_id)

        prepared = []
        for msg in item:
            try:
                vec = embed.embed([msg["content"]])[0]
            except Exception as e:
                print(f"embedding失败:{e}")
                vec = []
            prepared.append((msg,vec))
        try:
            n = self.store.llen(k_list)
            overflow = max(0,n - (max_turns - 1))
            stale_mid = []
            if overflow:
                for x in self.store.lrange(k_list,-overflow,-1):
                    try:
                        stale_mid.append(json.loads(x)["mid"])
                    except (ValueError, TypeError):
                        pass

            client = self.store.pipeline()
            for msg,vec in prepared:
                client.lpush(k_list,json.dumps(msg,ensure_ascii=False).encode())
                if len(vec) == 1024:
                    client.hset(k_vec,msg["mid"],np.asarray(vec,dtype = "<f4").tobytes())
            client.ltrim(k_list,0,max_turns - 1)
            if stale_mid:
                client.hdel(k_vec,*stale_mid)
            if expire_time:
                client.expire(k_list,expire_time)
                client.expire(k_vec,expire_time)
            client.execute()
            return True
        except Exception as e:
            print(f"存储出错{e}")
            return False

    def recent(self,session_id:str,limit:int = 20,user_id:str = "default")->list[dict]:
        """取最近 limit 轮，按时间正序（旧→新）返回。不做相关性过滤、不做分数门槛。"""
        k_list,_ = self.key(session_id,user_id)
        try:
            raw = self.store.lrange(k_list,0,limit - 1)      # index 0 是最新
        except Exception as e:
            print(f"查询出错{e}")
            return []
        if not raw:
            return []
        items = []
        for msg in reversed(raw):                            # lrange 拿的是 新→旧，翻成 旧→新
            try:
                items.append(json.loads(msg))
            except (ValueError, TypeError):
                continue
        return items

    def search_filter(self, m:dict | None = None,roles:set[str] |None = None,max_age:float | None = None,now:float|None = None)->bool:
        if roles and m.get("role") not in roles:
            return False
        if max_age is not None and (now or time.time()) - float(m.get("ts") or 0) > max_age:
            return False
        return True

    def search(self,session_id:str,query:str,limit:int = 10,user_id :str = "default",roles:set[str]|None = None,max_age:float|None = None)->list[dict]:
        k_list,k_vec = self.key(session_id,user_id)
        try:
            raw = self.store.lrange(k_list,0,SCAN - 1)
        except Exception as e:
            print(f"查询出错{e}")
            return []
        if not raw:
            return []

        items = []
        for msg in raw:
            try:
                items.append(json.loads(msg))
            except (ValueError, TypeError):
                continue

        items = [m for m in items if self.search_filter(m, roles=roles, max_age=max_age)]

        # 1. BM25
        scores = BM25([tokenize(m["content"]) for m in items]).score_all(tokenize(query))
        order_bm = sorted(range(len(items)),key = lambda i:-scores[i])
        cand = order_bm[:BM25_TOP]

        # 2.向量精排
        vec_score:dict[int,float] = {}
        try:
            qv = np.asarray(embed.embed([query])[0], dtype="<f4")
            if qv.size == embed.DIM and cand:
                blobs = self._raw.hmget(k_vec, [items[i]["mid"] for i in cand])   # ← 用 _raw
                for i, b in zip(cand, blobs):
                    if b:
                        v = np.frombuffer(b, dtype="<f4")
                        vec_score[i] = float(v @ qv)          # 向量已 L2 归一化
        except Exception:
            pass                                              # 向量不可用 → 退化为纯 BM25
        if vec_score:
            if max(vec_score.values()) < MIN_VEC_SIM:
                return []
            else:
                if not cand or scores[cand[0]] <= 0:
                    return []
        # ③ 融合：min-max 归一化后加权（BM25 与余弦量纲不同，必须先归一化）
        def norm(d: dict[int, float]) -> dict[int, float]:
            if not d:
                return {}
            lo, hi = min(d.values()), max(d.values())
            if hi > lo:
                return {k: (v - lo) / (hi - lo) for k,v in d.items()}
            elif hi == lo and lo == 0:
                return {k: 0.0 for k,_ in d.items()}
            else:
                return {k: 1.0 for k,_ in d.items()}

        nb, nv = norm({i: scores[i] for i in cand}), norm(vec_score)
        fused = {i: ALPHA * nb.get(i, 0.0) + (1 - ALPHA) * nv.get(i, 0.0) for i in cand}
        top = sorted(fused, key=lambda i: -fused[i])[:limit]

        out = [items[i] for i in top]
        out.sort(key = lambda m:m["ts"])                                    # 新→旧 转 旧→新
        return out

    @staticmethod
    def to_llm_messages(msgs: list[dict]) -> list[dict]:
        """剥掉内部字段（mid/ts），只给 LLM 认识的部分"""
        out = []
        for m in msgs:
            item = {"role": m["role"], "content": m["content"]}
            if m.get("name"):
                item["name"] = m["name"]
            out.append(item)
        return out