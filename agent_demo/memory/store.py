import pymysql
from contextlib import contextmanager
import struct,time
import os
from dotenv import load_dotenv
class StaleWrite(RuntimeError):
    pass

load_dotenv()

## 配置一下数据库的基础信息
DB_CONFIG = dict(
    host = "127.0.0.1",
    port = 3306,
    user = "root",
    password =os.getenv("PASSWORD"),
    database = "agent_memory",
    charset = "utf8mb4",
    cursorclass = pymysql.cursors.DictCursor,
)

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

class StaleWrite(RuntimeError):
    """要关闭的旧行已被别人关掉（并发/重复纠正）→ 调用方重新取候选再决定"""
    pass

class Store:
    def __init__(self,client):
        self.client = client
    # 2. 短期记忆
    def add_filter(self,message:str)->list[dict]:
        pass
    def add(self,client,message:list[dict]):
        pass
    def search_filter(self,query:str):
        pass
### 3. 长期记忆
    @staticmethod
    def _row(r:dict,with_embed:bool = False)->dict:
        """统一出口：默认不返回 embed """
        if r is None:
            return None
        r["embed"] = _unpack(r["embed"]) if (with_embed and r.get("embed")) else []
        return r
    def get_fact(self,fact_id:int,with_embed:bool = False)->dict|None:
        """取单条一排查/审计用，不受actives过滤"""
        with get_conn() as conn,conn.cursor() as cursor:
            cursor.execute(
                """
                SELECT id,user_id,content,kind,slot,source,scene,confidence,created_at,valid_from,valid_until,superseded_by,supersedes,superseded_at,time_precision,use_count,last_used,embed
                FROM facts WHERE id = %s 
                """,
                (fact_id,)
            )
            return self._row(cursor.fetchone(),with_embed)
    def earlier_facts(self,fact_id:int)->list[dict]:
        """沿着 supersedes 反向往前走：这条事实的前身链. 并且加了visited防止重复环+步数上限"""
        out,cur_id,visited = [],fact_id,set()
        while cur_id and cur_id not in visited and len(out) < 10:
            visited.add(cur_id)
            r = self.get_fact(cur_id)
            if r is None:
                break
            out.append(r)
            cur_id = r["supersedes"]
        return out

    def add_fact(self,content:str,embed:list[float],**meta)->int:
        with get_conn() as conn,conn.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO facts
                (user_id,content,kind,slot,source,scene,confidence,created_at,valid_from,time_precision,embed)
                VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                """,
                (
                meta.get("user_id","default"),content,
                meta.get("kind","fact"),meta.get("slot"),
                meta.get("source","user_explicit"),
                meta.get("scene"),meta.get("confidence",0.9),
                time.time(),
                meta.get("valid_from"),
                meta.get("time_precision","inferred"),
                _pack(embed)),
            )
            return cursor.lastrowid

    def active_facts(self,kinds:list[str]|None = None,user_id:str = "default",limit:int = 500,with_embed:bool = False)->list[dict]:
        cols = ("id,content,kind,slot,source,scene,confidence,created_at,valid_from,valid_until,superseded_by,supersedes,superseded_at,"
            "time_precision,use_count,last_used")
        if with_embed:
            cols += ",embed"
        sql = f"SELECT {cols} FROM facts WHERE user_id = %s AND valid_until IS NULL"
        params:list = [user_id]
        if kinds:
            ph = ",".join(["%s"]*len(kinds))
            sql += f" AND kind in ({ph})"
            params.extend(kinds)
        sql += " ORDER BY created_at DESC LIMIT %s"
        params.append(limit)
        with get_conn() as conn, conn.cursor() as cursor:
            cursor.execute(sql,params)
            return [self._row(r,with_embed) for r in cursor.fetchall()]

    def history_of(self,kind:str,slot:str,user_id:str = "default")->list[dict]:
        """一个槽位的完整变更史——按【现实时间】排，不按 id 排"""
        with get_conn() as conn, conn.cursor() as cursor:
            cursor.execute(
                """
                SELECT id,content,kind,slot,source,created_at,valid_from,valid_until,
                      superseded_by,supersedes,superseded_at,time_precision
                FROM facts
                WHERE user_id = %s AND kind = %s AND slot <=> %s
                ORDER BY COALESCE(valid_from,created_at) ASC,created_at ASC
                """,
                (user_id,kind,slot)
            )
            return cursor.fetchall()

    def as_of(self, ts: float, axis: str = "valid", kinds: list[str] | None = None,
              user_id: str = "default", limit: int = 200) -> list[dict]:
        """时间旅行，axis='valid'：当时【事实】是什么；axis='system'：当时【系统认为】是什么"""
        cols = ("id,content,kind,slot,source,created_at,valid_from,valid_until,"
            "superseded_by,supersedes,superseded_at,time_precision")
        if axis == "system":
            where = "created_at <= %s AND (superseded_at IS NULL OR superseded_at > %s)"
            time_params = (ts,ts)
            order = "ORDER BY created_at DESC"
        else:
            where = "(valid_from IS NULL OR valid_from <= %s) AND (valid_until IS NULL OR valid_until > %s)"
            time_params = (ts,ts)
            order = "ORDER BY COALESCE(valid_from,created_at) DESC"
        sql = f"SELECT {cols} FROM facts WHERE user_id = %s AND {where}"
        params = [user_id,*time_params]
        if kinds:
            ph = ",".join(["%s"]*len(kinds))
            sql += f" AND kind in ({ph})"
            params.extend(kinds)
        sql += f" {order} LIMIT %s"
        params.append(limit)
        with get_conn() as conn, conn.cursor() as cursor:
            cursor.execute(sql,params)
            return cursor.fetchall()

    def close_and_add(self,old_ids:list[int],content:str,embed:list[float],supersede_reason:str |None = None,**meta)->int:
        """一个事务内：插入新事实 + 关闭旧事实 + 打通双向指针。

        旧行的 valid_until 取【新事实的 valid_from】（现实时间线），
        而不是 time.time()（系统时间线）——这是双时间的核心，别写错。
        """
        now = time.time()
        valid_from = meta.get("valid_from")
        end = valid_from if( valid_from is not None and meta.get("time_precision") == "exact") else now
        with get_conn() as conn,conn.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO facts
                (user_id,content,kind,slot,source,scene,confidence,created_at,valid_from,time_precision,embed)
                VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                """,
                (
                    meta.get("user_id","default"),
                    content,
                    meta.get("kind","fact"),
                    meta.get("slot"),
                    meta.get("source","user_explicit"),
                    meta.get("scene"),
                    meta.get("confidence",0.9),
                    now,valid_from,
                    meta.get("time_precision","inferred"),
                    _pack(embed),
                )
            )
            new_id = cursor.lastrowid

            if old_ids:
                ph = ",".join(["%s"]*len(old_ids))
                cursor.execute(
                    f"""
                    UPDATE facts
                        SET valid_until = %s,
                            superseded_at = %s,
                            superseded_by = %s
                        WHERE id IN ({ph}) AND valid_until IS NULL
                    """,
                    (
                        end,now,new_id,*old_ids,
                    )
                )
                if cursor.rowcount != len(old_ids):
                    raise StaleWrite(
                        f"有 {len(old_ids) - cursor.rowcount} 条旧记忆已被其它操作失效，"
                        f"请重新取候选再决定（new_id={new_id} 已回滚）"
                    )
            for oid in old_ids:
                cursor.execute("UPDATE facts SET supersedes = %s WHERE id = %s", (oid, new_id))
            return new_id


    def touch(self,fact_ids:list[int])->None:
        if not fact_ids:
            return
        ph = ",".join(["%s"]*len(fact_ids))
        with get_conn() as conn,conn.cursor() as cursor:
            cursor.execute(
                f"""
                UPDATE facts SET use_count = use_count + 1 ,last_used = %s
                WHERE id IN ({ph})""",
                (time.time(),*fact_ids),
            )


if __name__ == "__main__":
    s = Store()
    print("active:", [(f["id"], f["kind"], f["slot"], f["content"][:12]) for f in s.active_facts()])
    print("history(address):", s.history_of("identity", "address"))
    print("as_of(now, valid):", len(s.as_of(time.time(), "valid")))
    print("get_fact(1):", s.get_fact(1)["content"])
    print("earlier_facts(1):", [x["id"] for x in s.earlier_facts(1)])