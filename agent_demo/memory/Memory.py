import time,json

from httpcore2 import WriteError
from datetime import datetime,timezone
from . import embed              # ← 原来是 import time,embed
from .store import Store, StaleWrite  # ← 原来是 from store import Store

HALF_LIFE_DAYS = 45.0
DECAY_FLOOR = 0.30
MIN_BEST_SIM = 0.50
TOKEN_BUDGET = 600
REL_FLOOR = 0.75
query_mapping = [
    (["住", "在哪", "哪里", "城市", "地址", "老家"], ["identity"]),
    (["喜欢", "讨厌", "偏好", "习惯", "不喜欢", "别给我"], ["preference", "constraint"]),
    (["在学", "准备", "打算", "目标", "计划", "忙什么"], ["goal"]),
    (["叫什么", "是谁", "多大", "做什么工作", "身份"], ["identity"]),
]
VALID_KINDS = {"identity", "preference", "goal", "constraint", "event", "fact"}
SINGLE_VALUE_SLOTS = frozenset(["address", "name", "job", "age", "hometown", "city"])
DEFAULT_SLOT = "general"
SLOT_ENUM = ["address","name","job","age","hometown","city","dietary","hobby","dislike","goal","general"]
SOURCE_RANK = {"user_correction":3,"user_explicit":2,"agent_inferred":0}

def slot_of(kind:str,slot:str | None = None)->str:
    """兜底：如果没有传入slot 就用kind兜底。"""
    return (slot or "").strip().lower() or (kind or DEFAULT_SLOT)

def can_supersede(old:dict,new_source:str)->bool:
    """两条资格判定：1.必须是单值槽位 2.source_rank 必须更大"""
    if old.get("slot") not in SINGLE_VALUE_SLOTS:
        return False
    else:
        return SOURCE_RANK.get(new_source,2) >= SOURCE_RANK.get(old.get("source"),2)

def fmt_time(ts:float)->str:
    """给 LLM 看的时间要用相对量：'3天前' 远比 1789985300 有用"""
    if not ts:                      # None / 0 都视为未知，否则会算成"56.8年前"
        return "未知"
    now = time.time()
    age = (now - ts)/86400
    if age < 1/24:
        return "刚刚"
    elif age < 1:
        return f"{age * 24:.0f}小时前"
    elif age < 30:
        return f"{age:.0f}天前"
    elif age < 365:
        return f"{age / 30:.1f}月前"
    else:
        return f"{age/365:.1f}年前"

def _parse_ts(s:str)->float | None:
    s = s.strip()
    if not s:
        return None
    try:                                     # 纯数字 = unix 秒
        return float(s)
    except ValueError:
        pass
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d", "%Y/%m/%d"):
        try:
            return datetime.strptime(s, fmt).replace(tzinfo=timezone.utc).timestamp()
        except ValueError:
            continue
    return None

def detect_kind(query:str):
    kinds = set()
    for mapping,filed in query_mapping:
        if any(kw in query for kw in mapping):
            kinds.update(filed)
    return list(kinds) if kinds else None

class Memory:
    def __init__(self,user_id: str = "default", scene: str | None = None):
        self.store = Store()
        self.user_id = user_id
        self.scene = scene
    @staticmethod
    def _out(rows:list)->list:
        keys = ("id","content","kind","slot","created_at","valid_from","use_count","score","sim","error")
        return [{k: r[k] for k in keys if k in r}for r in rows]

    def recall(self,query:str,kind:str | None = None,k=3,as_of : str | None = None,axis:str = "valid"):
        """检索已记住的关于用户的信息。当问题涉及用户的住址、偏好、目标、正在做的事时必须先调用。
        query: 用自然语言描述要找什么，例如'用户住在哪里'
        kind: identity/preference/goal/constraint/event/fact，不确定就不传（会自动判断）
        as_of: 用户问【过去某时刻】的情况时传 ISO 日期，如 '2026-05-01'
        axis: valid=当时事实是什么（默认）；system=当时系统认为是什么
        """
        k = int(k or 3)
        qv = embed.fetch(query)[0]
        if not qv:
            return []
        if as_of:
            ts = _parse_ts(as_of)
            if ts is None:
                return [{"error":f"as_of 格式无法正确解析：{as_of}，请用'YYYY-MM--DD'"}]
            kinds = [kind] if kind else None
            return self._out(self.store.as_of(ts,axis = axis,kinds = kinds,user_id =self.user_id,limit = 20)[:k])
        kinds = [kind] if kind else detect_kind(query)
        facts = self.store.active_facts(kinds = kinds,user_id = self.user_id,with_embed = True)
        now,scored = time.time(),[]
        for f in facts:
            v = f.get("embed") or []
            if len(v) != len(qv):
                continue
            sim  = sum(a * b for a,b in zip(qv,v))
            if sim <= 0:
                continue
            age = (now - f["created_at"]) / 86400
            decay = 0.5 ** (age / HALF_LIFE_DAYS)
            scored.append((sim * (DECAY_FLOOR + (1 - DECAY_FLOOR) * decay), sim, f))
        if not scored:
            return []
        scored.sort(key=lambda x: x[0], reverse=True)
        if max(s for _,s,_ in scored) < MIN_BEST_SIM:
            return []

        picked,used = [],0
        for score,sim,f in scored:
            cost = len(f["content"])
            if used + cost > TOKEN_BUDGET:
                continue
            used += cost
            picked.append({**f,"score":round(score,4),"sim":round(sim,4)})
            if len(picked) >= k:
                break
        # 记录一下记忆召回的次数
        id_list = []
        if picked:
            self.store.touch([f["id"] for f in picked])
        return self._out(picked)

    def remember(self,content:str,kind:str = "fact",slot :str | None = None)->str:
        """记录关于用户的长期信息。当用户透露身份、偏好、目标、约束，或明确要求记住某事时调用。
        content: 一句完整的事实陈述，不要用代词，例如'用户常住上海'
        kind: identity/preference/goal/constraint/event/fact
        slot: 这条事实的谓词槽位，必填。address=住址 name=姓名 job=职业 age=年龄
              hometown=老家 dietary=口味忌口 hobby=爱好 dislike=讨厌 goal=目标 general=其它
        """
        content = content.strip()
        if len(content) < 4:
            return "内容过短，未记录。"
        if kind not in VALID_KINDS:  # ← 新增：非法值兜底，不报错
            kind = "fact"
        slot = slot_of(kind = kind,slot = slot)

        vec = embed.fetch(content)[0]
        if not vec:
            return "向量服务异常，本次未记录"

        cands = self.store.active_facts(kinds = [kind],user_id = self.user_id,with_embed = True)
        scored = []
        for f in cands:
            v = f["embed"]
            if not v or len(v) != len(vec):
                continue
            sim = sum(a*b for a,b in zip(vec,v))
            if f.get("slot") == slot:
                scored.append((sim,f))
        scored.sort(key = lambda x : x[0],reverse = True)
        scored = scored[:5]

        to_close = [f["id"] for _,f in scored if can_supersede(f,"user_explicit")]
        #1.有矛盾的情况
        if to_close:
            try:
                self.store.close_and_add(to_close,content,vec,user_id = self.user_id,kind = kind,slot = slot,scene = self.scene,source = "user_explicit",valid_from = None,time_precision="inferred")
            except StaleWrite as e:
                self.store.add_fact(content, vec, user_id=self.user_id, kind=kind,
                                    slot=slot, scene=self.scene, source="user_explicit")
                return f"已记住(旧记忆状态已变化:{e}):{content}"
            return f"已记住(并更新了{len(to_close)}条旧消息:{content})"
        #2. 无矛盾的情况
        self.store.add_fact(content,vec,kind = kind,source = "user_explicit",slot = slot,scene =self.scene,valid_from = None,time_precision = "inferred",user_id = self.user_id)
        return f"已记住:{content}"

    def history(self, kind: str, slot: str) -> list:
        """某槽位的完整变更史，用于回答'我住址变过几次'以及排查召回冲突"""
        return self._out(self.store.history_of(kind, slot, user_id=self.user_id))






