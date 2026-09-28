"""memory_test.py —— 检索排序验证。（注意：本文件需在 memory/ 目录下运行）"""
import time

import embed
from Memory import Memory
from store import _pack, get_conn      # ← get_conn 是模块级函数，从模块导入

DAY = 86400


def add_backdated(store, text, vec, kind, source, age_days):
    """入库并回填 created_at。

    Store.add_fact() 内部写死 time.time()，无法指定创建时间；
    而验证时间衰减必须造出"3天前/120天前"的记忆，所以这里单独走 SQL。
    生产代码用 add_fact 即可，这个函数只服务于测试。
    """
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            """INSERT INTO facts (user_id,content,kind,source,confidence,created_at,embed)
               VALUES (%s,%s,%s,%s,%s,%s,%s)""",
            ("default", text, kind, source, 0.9,
             time.time() - age_days * DAY, _pack(vec)),
        )


def main():
    m = Memory()

    # ── 1. 清空旧数据（里面有 [0.1]*1024 的假向量）──
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute("TRUNCATE TABLE facts")
    print("facts 表已清空")

    # ── 2. 灌入 5 条真实记忆，年龄各不相同 ──
    seed = [
        ("用户常住上海",                     "identity",   "user_explicit", 120),
        ("用户已从上海搬到北京",               "identity",   "user_explicit",   3),
        ("用户喜欢简洁的回答，不喜欢长篇大论",   "preference", "user_explicit",  30),
        ("用户正在学习 Agent 开发",            "goal",       "user_explicit",   5),
        ("用户不喜欢吃辣",                    "preference", "user_explicit",  60),
    ]
    vecs = embed.fetch([s[0] for s in seed])
    for (text, kind, src, age), v in zip(seed, vecs):
        add_backdated(m.store, text, v, kind, src, age)
    print(f"已写入 {len(seed)} 条记忆\n")

    # ── 3. 逐个问题检索 ──
    for q in ["我住在哪里？", "我最近在学什么？", "给我讲讲这个原理", "今天股市怎么样","我有什么需要注意的"]:
        res = m.recall(q)                       # 只调一次，别调两遍
        print(f">>> {q}")
        if not res:
            print("    （无记忆越过阈值 → 不注入）")
        for f in res:
            print(f"    score={f['score']:.4f} sim={f['sim']:.4f}  {f['content']}")
        print()


if __name__ == "__main__":
    main()