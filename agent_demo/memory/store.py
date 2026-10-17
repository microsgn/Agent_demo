import asyncio
import pymysql
from contextlib import contextmanager
import struct,time
import os
from pathlib import Path
from dotenv import load_dotenv
from .embed import embed,EMBED_MODEL
from datetime import datetime
from models.Memory_facts import MemoryItem,MemoryBatch,FactsResponse
from models.DB_Config import DB_CONFIG,DB_field
from .retrieve import retrieve
class StaleWrite(RuntimeError):
    pass

load_dotenv()
from langchain.chat_models import init_chat_model
llm = init_chat_model(
    api_key = os.getenv("DEEPSEEK_API_KEY"),
    model = "deepseek-v4-pro",
    model_provider = "openai",
    base_url = os.getenv("DEEPSEEK_URL"),
    extra_body={"thinking": {"type": "disabled"}}
)
model = llm.with_structured_output(MemoryBatch,method="function_calling")
## 配置一下数据库的基础信息
kind_map = [
    "identity",
    "address",
    "preference",
    "experience",
    "default"
]
### 利用with方式进行访问，实现with结束即关闭，避免脏状态
@contextmanager
def get_conn():
    conn = pymysql.connect(**DB_CONFIG)
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()

def _pack(vec:list[float])->bytes:
    return struct.pack(f"<{len(vec)}f", *vec)
def _unpack(blob:bytes)->list[float]:
    if len(blob) % 4 != 0:
        raise ValueError(f"BLOB 长度 {len(blob)} 不是 4 的倍数，数据已损坏")
    return list(struct.unpack(f"<{len(blob)//4}f",blob))
def session_id(user_id:str)->str:
    ts = datetime.now().strftime("%Y%m%d%H%M%S")
    return f"{ts}-{user_id}"
def load_prompt()->str:
    base_dir = Path(__file__).resolve().parent.parent
    path = base_dir / "Prompts"/"LLM初筛.md"
    if not path.exists():
        raise FileNotFoundError(f"Prompt 文件不存在: {path}")
    return path.read_text(encoding="utf-8").strip()

class StaleWrite(RuntimeError):
    """要关闭的旧行已被别人关掉（并发/重复纠正）→ 调用方重新取候选再决定"""
    pass

class Store:
    def __init__(self):
        self.review = []
    def add_filter(self,texts:list[str],user_id:str)->list[dict]:
        """
        将文本传给LLM，让LLM判断是否需要存入长期记忆
        """
        system_prompt = load_prompt()
        out = []
        for text in texts:
            old_message = retrieve(text,user_id)
            input_text = {
                "next_text":text,
                "related_memories":old_message,
            }
            messages = [
                {"role":"system","content":system_prompt},
                {"role":"user","content":f"这是新的需要判断是否存入长期记忆的消息以及和可能他相关的记忆{input_text}"},
        ]
            response = model.invoke(messages)
            for item in response.items:
                d = item.model_dump()
                if d["should_store"] == False:
                    continue
                if d["is_update"] == True:
                    d["embedding"] = _pack(embed([item.content])[0])
                    self.review.append(d)
                    continue
                d["embedding"] = _pack(embed([item.content])[0])  # 对 item 自己的 content 嵌入
                out.append(d)
        return out

    def mannul_filter(self,texts:list[str],user_id:str):
        if not self.review:
            print("暂无待复核记忆。")
            return
        message = self.add_filter(texts,user_id)
        self.add(message, user_id, session_id(user_id))
        content_list = [d["content"] for d in self.review]
        print(f"待复核列表{content_list}")
        for msg in self.review.copy():
            print(f"当前复核{msg["content"]}")
            ans = input("接受:A  拒绝:R  返回聊天:Q 否则跳过")
            if ans.strip().upper() == "A":
                self.add([msg],user_id,session_id(user_id))
                idx = self.review.index(msg)
                self.review.pop(idx)
                continue
            elif ans.strip().upper() == "R":
                idx = self.review.index(msg)
                self.review.pop(idx)
                continue
            elif ans.strip().upper() == "Q":
                break
            else:
                continue
        print(f"剩余复核列表{[d["content"] for d in self.review]}")

    def add(self,texts:list[dict],user_id:str,session_id:str,map:list = kind_map):
        sql1 = """
        insert into facts (user_id,session_id,content,kind,embedding,embed_model,is_update,conflict_with) 
        values (%s,%s,%s,%s,%s,%s,%s,%s)
        """
        sql2 = """
        update facts set status = 'superseded' where id = %s
        """
        with get_conn() as conn, conn.cursor() as cursor:
            for msg in texts:
                text = msg
                if "kind" not in text.keys() or text["kind"] not in map:
                    continue
                params = (user_id,session_id,text["content"],text["kind"],text["embedding"],EMBED_MODEL,text["is_update"],text["conflict_with"])
                cursor.execute(sql1,params)
                if text["conflict_with"] is not None :
                    params = (text["conflict_with"],)
                    cursor.execute(sql2,params)
            cursor.close()
    def process_memory(self,texts:list[str],user_id:str,session_id:str)->None:
        candidates = self.add_filter(texts,user_id)
        if candidates:
            self.add(candidates,user_id,session_id)
        if self.review:
            print(f"有 {len(self.review)} 条待复核记忆，输入 /review 查看。")

    def search(self,messages:list[str],user_id:str)->list[dict]:
        output = []
        for i,msg in enumerate(messages):
            response = retrieve(msg,user_id)
            out = FactsResponse(index = i,response = response)
            output.append(out.model_dump())
        return output

if __name__ == "__main__":
    texts = [
        "我叫jason hua",  # identity
        "我住在哈尔滨",  # address
    ]
    s = Store()
    # print("开始初筛：只有更新候选会进入人工复核。")
    # asyncio.run(s.mannul_filter(texts, user_id="001"))
    # print("复核结束，仍待复核：", [item["content"] for item in s.review])

    query = ["我是谁？","我住哪？"]
    resp = s.search(query,user_id = "001")
    for msg in resp:
        print(msg["response"])
