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
embed_model = os.getenv("OLLMA_MODEL")
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
VECTOR_TOP = 50
RRF_K = 60
ALPHA = 0.4
MIN_VEC_SIM = 0.50      # 最低余弦相似度，低于此值视为"都不相关"。与 Memory.MIN_BEST_SIM 同尺

class ShortTermMemory():
    def __init__(self,host:str = HOST,port:str = PORT,prefix:str = key_prefix):
        self.store = redis.Redis(host = host,port = int(port),decode_responses = True)
        self._raw = redis.Redis(host=host, port=int(port), decode_responses=False)
        self.prefix = prefix

    def key(self,session_id:str,user_id:str = "default",embed_model:str |None = embed_model)->tuple[str,str]:
        base = f"{self.prefix}:{session_id}:{user_id}#{embed_model}-"
        return f"{base}turns",f"{base}vec"

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
        k_list,k_vec = self.key(session_id = session_id,user_id = user_id)

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
            overflow = max(0,n - (max_turns - len(item)))
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
                if len(vec) == embed.DIM:
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
        k_list,_ = self.key(session_id = session_id,user_id = user_id)
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
        """两路独立召回，按照RRF融合，返回结果按时间旧到新排列"""
        if limit < 0:
            raise ValueError(f"limit必须大于0")
        if not query.strip():
            return []

        k_list,k_vec = self.key(session_id = session_id,user_id = user_id)
        try:
            raw = self.store.lrange(k_list,0,SCAN - 1)
        except Exception as e:
            print(f"查询出错{e}")
            return []
        if not raw:
            return []
## 两路独立召回共用同一过滤的消息，构建 key:mid -> value:msg 来进行映射连接
        now = time.time()
        by_mid :dict[str,dict] = {}
        for row in raw:
            msg = json.loads(row)
            if self.search_filter(msg,roles=roles,max_age=max_age,now=now):
                by_mid.setdefault(msg["mid"],msg)

        items = list(by_mid.values())
        if not items:
            return []

        # 1. BM25
        scores = BM25([tokenize(msg["content"]) for msg in items]).score_all(tokenize(query))
        order_bm = sorted((i for i,score in enumerate(scores) if score > 0), key=lambda i: scores[i],reverse=True)
        cand = order_bm[:BM25_TOP]

        # 2.向量独立召回
        def unit_vector(values)->np.ndarray:
            vec = np.asarray(values,dtype = np.float64)
            if vec.shape != (embed.DIM,) or not np.isfinite(vec).all():
                raise ValueError(f"向量维度错误或包含非有限值")
            norm = float(np.linalg.norm(vec))
            if not np.isfinite(norm) or norm <= 0:
                raise ValueError("向量范数必须大于0")
            return vec / norm

        query_vectors = embed.embed([query])
        if len(query_vectors) != 1:
            raise ValueError("查询 Embedding 数量应为1")
        qv = unit_vector(query_vectors[0])
        blobs = self._raw.hmget(k_vec,[msg["mid"] for msg in items])
        vec_score:dict[int,float] = {}
        for i,blob in zip(range(len(items)),blobs,strict=True):
            if blob is None:
                continue
            vec = unit_vector(np.frombuffer(blob,dtype = "<f4"))
            sim = float(np.clip(vec @ qv,-1.0,1.0))
            if sim < MIN_VEC_SIM:
                continue
            vec_score[i] = sim
        vec_order = sorted(vec_score,key = lambda i:vec_score[i],reverse = True)[:VECTOR_TOP]

## 3. 两路联合，采用RRF
        fused : dict[str,float] = {}
        for order in (cand,vec_order):
            for rank, i in enumerate(order,start = 1):
                mid = items[i]["mid"]
                fused[str(mid)] = fused.get(str(mid),0.0) + 1.0 / (RRF_K + rank)

        top_mids = sorted(fused,key = fused.get,reverse = True)[:limit]
        result = [by_mid[mid] for mid in top_mids]
        result.sort(key = lambda msg:msg["ts"])
        return result


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