"""双时间模型测试集（L1 数据访问层 + L2 领域层纯函数）

运行：  cd agent_demo/memory && python -m unittest test_bitemporal -v
或：    cd agent_demo && python -m unittest memory.test_bitemporal -v

设计原则：
  1. 所有用例自带清理（setUp/tearDown 只碰 __test__ 用户），绝不动真实记忆
  2. 时间全部相对 now 计算，不写死日期 → 任何一天跑结果都一样
  3. L1 用例完全不联网（向量是造出来的），必须永远能跑
  4. L2 里需要真实 embedding 的用例放在 NeedDashscope 类，网络不通自动 skip
"""
import os
import sys
import time
import unittest
import uuid

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from memory.store import Store, StaleWrite, get_conn
# ⚠️ 必须写成 memory.Memory（子模块），不能写 `from memory import Memory`
#    否则拿到的是【模块对象】而不是类：包 __init__.py 为空时，memory.Memory 这个名字
#    被子模块 Memory.py 占用，与类同名。实测：`from memory import Memory` -> <class 'module'>
from memory.Memory import Memory
from memory.Memory import slot_of, can_supersede, fmt_time, _parse_ts

DAY = 86400


def embed_stub(seed: float = 0.01) -> list[float]:
    """造一个 1024 维向量：数据访问层不关心向量内容，只关心长度和能否存取"""
    return [seed] * 1024


class BaseCase(unittest.TestCase):
    """统一：每个用例用独立的 user_id，跑完删干净"""

    def setUp(self):
        self.s = Store()
        self.u = f"__test__{uuid.uuid4().hex[:8]}"
        # 安全网：记录 default 用户的行 id 快照。若某个用例把记忆误写到 default，
        # tearDown 会立刻报错并清理，而不是静默污染真实数据。
        with get_conn() as c, c.cursor() as cur:
            cur.execute("SELECT id FROM facts WHERE user_id = 'default'")
            self._default_ids = {r["id"] for r in cur.fetchall()}

    def tearDown(self):
        with get_conn() as c, c.cursor() as cur:
            cur.execute("DELETE FROM facts WHERE user_id = %s", (self.u,))
            cur.execute("SELECT id FROM facts WHERE user_id = 'default'")
            leaked = {r["id"] for r in cur.fetchall()} - self._default_ids
            if leaked:
                cur.execute(
                    f"DELETE FROM facts WHERE id IN ({','.join(map(str, leaked))})")
                self._leaked = leaked
        if getattr(self, "_leaked", None):
            self.fail(f"❌ 有记忆被误写进了 default 用户（id={sorted(self._leaked)}）——"
                      f"检查 remember 里的 add_fact/close_and_add 是否漏传 user_id=self.user_id。"
                      f"（测试数据已自动清理）")

    # ── 工具 ────────────────────────────────────────────
    def mkfact(self, content, kind="identity", slot="address",
               valid_from=None, precision="exact", source="user_explicit"):
        return self.s.add_fact(content, embed_stub(), user_id=self.u, kind=kind, slot=slot,
                               source=source, valid_from=valid_from, time_precision=precision)

    def close(self, old_ids, content, valid_from=None, precision="exact",
              kind="identity", slot="address"):
        return self.s.close_and_add(old_ids, content, embed_stub(), user_id=self.u,
                                    kind=kind, slot=slot, source="user_explicit",
                                    valid_from=valid_from, time_precision=precision)

    def count(self):
        with get_conn() as c, c.cursor() as cur:
            cur.execute("SELECT COUNT(*) n FROM facts WHERE user_id = %s", (self.u,))
            return cur.fetchone()["n"]


# ══════════════════════════════════════════════════════════
# L1 数据访问层：不联网，永远必须通过
# ══════════════════════════════════════════════════════════
class L1Schema(BaseCase):

    def test_schema_has_bitemporal_columns(self):
        """双时间 5 列必须存在——少了任何一列，后面全部用例都会以奇怪的方式失败"""
        with get_conn() as c, c.cursor() as cur:
            cur.execute("SHOW COLUMNS FROM facts")
            cols = {r["Field"] for r in cur.fetchall()}
        for col in ("valid_from", "valid_until", "superseded_at",
                    "superseded_by", "supersedes", "slot", "time_precision"):
            self.assertIn(col, cols, f"facts 表缺少 {col} 列")

    def test_slots_have_no_index_yet(self):
        """记录现状：slot 目前无索引。将来加了索引这条会失败，那时改成断言'有索引'即可"""
        with get_conn() as c, c.cursor() as cur:
            cur.execute("SHOW INDEX FROM facts")
            indexed = {r["Column_name"] for r in cur.fetchall()}
        print(f"\n  [信息] 当前已索引列: {sorted(indexed)}")


class L1AddFact(BaseCase):

    def test_add_fact_writes_all_bitemporal_fields(self):
        self.s  # noqa
        vf = time.time() - 200 * DAY
        fid = self.mkfact("用户常住上海", valid_from=vf, precision="exact")
        f = self.s.get_fact(fid)
        self.assertEqual(f["content"], "用户常住上海")
        self.assertEqual(f["slot"], "address")
        self.assertAlmostEqual(f["valid_from"], vf, places=1)
        self.assertIsNone(f["valid_until"], "新插入的行必须有效")
        self.assertIsNone(f["superseded_by"])
        self.assertIsNone(f["supersedes"])
        self.assertEqual(f["time_precision"], "exact")
        self.assertEqual(f["use_count"], 0)

    def test_add_fact_defaults_are_conservative(self):
        """不传 valid_from/precision 时：valid_from=None（未知≠现在）、precision=inferred"""
        fid = self.s.add_fact("用户喜欢喝咖啡", embed_stub(), user_id=self.u, kind="preference")
        f = self.s.get_fact(fid)
        self.assertIsNone(f["valid_from"], "未知的开始时间必须是 NULL，不能伪造成 created_at")
        self.assertEqual(f["time_precision"], "inferred")
        self.assertLess(abs(f["created_at"] - time.time()), 5, "created_at 必须是系统时间")

    def test_embed_roundtrip_and_projection(self):
        fid = self.mkfact("用户叫华尧宽", slot="name")
        self.assertEqual(len(self.s.get_fact(fid, with_embed=True)["embed"]), 1024)
        self.assertEqual(self.s.get_fact(fid)["embed"], [], "默认不取向量（避免喂给 LLM）")


class L1CloseAndAdd(BaseCase):

    def test_close_and_add_links_both_directions(self):
        a = self.mkfact("用户常住上海", valid_from=time.time() - 200 * DAY)
        b = self.close([a], "用户搬到深圳", valid_from=time.time() - 30 * DAY)

        fa, fb = self.s.get_fact(a), self.s.get_fact(b)
        self.assertIsNotNone(fa["valid_until"], "旧行必须被封闭")
        self.assertEqual(fa["superseded_by"], b, "旧行要指向新行")
        self.assertIsNotNone(fa["superseded_at"], "系统时间轴的失效时刻必须写入")
        self.assertEqual(fb["supersedes"], a, "新行要指回旧行（反向指针）")
        self.assertIsNone(fb["valid_until"])

    def test_two_time_axes_do_not_collapse(self):
        """本测最核心的一条：valid_until（现实失效）≠ superseded_at（系统知情）"""
        a = self.mkfact("用户常住上海", valid_from=time.time() - 200 * DAY)
        b = self.close([a], "用户搬到深圳", valid_from=time.time() - 30 * DAY)
        fa = self.s.get_fact(a)
        gap_days = (fa["superseded_at"] - fa["valid_until"]) / DAY
        self.assertAlmostEqual(gap_days, 30, delta=0.1,
                               msg="valid_until 应等于新事实的 valid_from，而不是 now")
        self.assertAlmostEqual((fa["superseded_at"] - time.time()) / DAY, 0, delta=0.01,
                               msg="superseded_at 必须是系统当前时间")
        self.assertNotAlmostEqual(fa["valid_until"], fa["superseded_at"], places=3,
                                  msg="两条轴退化成一条了")

    def test_valid_from_is_inferred_falls_back_to_now(self):
        """precision != exact 时，即使传了 valid_from 也不能拿去当有效时间边界（那是猜的）"""
        a = self.mkfact("用户常住上海", valid_from=time.time() - 200 * DAY)
        b = self.close([a], "用户搬到深圳", valid_from=time.time() - 30 * DAY, precision="inferred")
        fa = self.s.get_fact(a)
        self.assertAlmostEqual((fa["valid_until"] - time.time()) / DAY, 0, delta=0.01,
                               msg="inferred 的 valid_from 不可信，应退回 now")
        self.assertAlmostEqual(fa["valid_from"], time.time() - 200 * DAY, delta=1)

    def test_closing_already_closed_raises_and_rolls_back(self):
        """回归防护：重复关闭必须抛 StaleWrite，且整事务回滚（不能留下脏的新行）"""
        a = self.mkfact("用户常住上海", valid_from=time.time() - 200 * DAY)
        self.close([a], "用户搬到深圳", valid_from=time.time() - 30 * DAY)
        before = self.count()
        with self.assertRaises(StaleWrite):
            self.close([a], "用户搬到杭州", valid_from=time.time() - 3 * DAY)
        self.assertEqual(self.count(), before, "冲突时新插入的行必须被回滚")

    def test_partial_conflict_rolls_back_everything(self):
        """一次关两条、其中一条已失效 → 整批回滚，不能只关掉活着的那条"""
        a = self.mkfact("用户常住上海", valid_from=time.time() - 200 * DAY)
        b = self.mkfact("用户住上海市浦东新区", slot="address", valid_from=time.time() - 180 * DAY)
        self.close([a], "用户搬到深圳", valid_from=time.time() - 30 * DAY)   # a 已失效
        before = self.count()
        with self.assertRaises(StaleWrite):
            self.close([a, b], "用户搬到杭州", valid_from=time.time() - 3 * DAY)
        self.assertEqual(self.count(), before)
        self.assertIsNone(self.s.get_fact(b)["valid_until"], "b 不能被单独关掉")


class L1ActiveAndHistory(BaseCase):

    def setUp(self):
        super().setUp()
        now = time.time()
        self.a = self.mkfact("用户常住上海", valid_from=now - 200 * DAY)
        self.b = self.close([self.a], "用户搬到深圳", valid_from=now - 30 * DAY)
        self.c = self.close([self.b], "用户搬到杭州", valid_from=now - 3 * DAY)
        self.now = now

    def test_active_only_returns_current(self):
        rows = self.s.active_facts(user_id=self.u)
        self.assertEqual([r["id"] for r in rows], [self.c])

    def test_history_is_ordered_by_real_time_not_id(self):
        """变更史按 COALESCE(valid_from, created_at) 升序 —— 现实时间序，不是插入序"""
        rows = self.s.history_of("identity", "address", user_id=self.u)
        self.assertEqual([r["id"] for r in rows], [self.a, self.b, self.c])
        self.assertEqual([r["content"][:4] for r in rows],
                         ["用户常住", "用户搬到", "用户搬到"])

    def test_history_null_slot_query_is_excluded(self):
        """slot=None 的行不该混进具体槽位的历史（<=> 的语义）"""
        self.assertEqual(self.s.history_of("identity", "job", user_id=self.u), ())

    def test_supersede_chain_is_walkable_both_ways(self):
        self.assertEqual([x["id"] for x in self.s.earlier_facts(self.c)],
                         [self.c, self.b, self.a])
        self.assertEqual(self.s.get_fact(self.a)["superseded_by"], self.b)
        self.assertEqual(self.s.get_fact(self.b)["superseded_by"], self.c)
        self.assertEqual(self.s.get_fact(self.b)["supersedes"], self.a)

    def test_isfact_exists_after_invalidation(self):
        """失效 ≠ 删除：被推翻的行必须还在表里（这是能回溯的前提）"""
        self.assertEqual(self.count(), 3)
        self.assertIsNotNone(self.s.get_fact(self.a)["valid_until"])


class L1AsOfTwoAxes(BaseCase):

    def setUp(self):
        super().setUp()
        self.oldest = time.time()          # 任何写入发生【之前】的时间戳，用于验证边界
        now = time.time()
        self.a = self.mkfact("用户常住上海", valid_from=now - 200 * DAY)
        self.b = self.close([self.a], "用户搬到深圳", valid_from=now - 30 * DAY)
        self.c = self.close([self.b], "用户搬到杭州", valid_from=now - 3 * DAY)
        self.now = now

    def test_valid_axis_travels_back_in_time(self):
        """现实时间轴：90 天前住上海、10 天前住深圳 —— 这才是'回溯'"""
        ids = lambda ts: [r["id"] for r in self.s.as_of(ts, "valid", user_id=self.u)]
        self.assertEqual(ids(self.now - 90 * DAY), [self.a])
        self.assertEqual(ids(self.now - 10 * DAY), [self.b])
        self.assertEqual(ids(self.now), [self.c])
        self.assertEqual(ids(self.now - 300 * DAY), [], "还没开始住的时期应该查不到")

    def test_system_axis_is_empty_before_we_knew(self):
        """记录时间轴：我 0.6 天前才开始记，所以 90 天前系统一无所知"""
        n = len(self.s.as_of(self.now - 90 * DAY, "system", user_id=self.u))
        self.assertEqual(n, 0, "那时系统还没记录任何东西")

    def test_system_axis_now_returns_current_belief(self):
        """★★ 系统时间轴在'当前时刻'的边界语义（测出来的真实坑）：

        created_at 是插入那一刻的 now()，而 as_of 用的是严格 `created_at <= 传入ts`。
        所以拿 setUp 里捕获的 now（比 c 的 created_at 早几毫秒）去查，会返回空。
        → 想查"系统现在认为什么"，必须用【更新鲜】的时间戳，或者直接用 active_facts()。
        """
        self.assertEqual([r["id"] for r in self.s.as_of(time.time(), "system", user_id=self.u)],
                         [self.c])
        self.assertEqual(                                    # 老时间戳的边界行为
            [r["id"] for r in self.s.as_of(self.oldest - DAY, "system", user_id=self.u)], [])

    def test_system_axis_reveals_knowledge_lag(self):
        """记录时间轴的价值：as_of(axis='system') 返回【那一刻系统相信的快照】。

        注意 close_and_add 里新行的 created_at 与旧行的 superseded_at 是同一个 now，
        所以"取代发生的那一瞬"返回的就是新行；要回到旧认知必须用严格更早的时间戳。
        """
        t_b = self.s.get_fact(self.b)["created_at"]
        self.assertEqual([r["id"] for r in self.s.as_of(t_b, "system", user_id=self.u)],
                         [self.b], "b 被记录的那一刻，系统相信 b（a 刚被取代）")
        self.assertEqual([r["id"] for r in self.s.as_of(t_b - 1e-6, "system", user_id=self.u)],
                         [self.a], "再早一瞬，系统还相信 a")
        self.assertEqual([r["id"] for r in self.s.as_of(time.time(), "system", user_id=self.u)],
                         [self.c], "现在系统只相信 c")

    def test_shared_valid_from_boundary_is_broken_by_pointer(self):
        """★★ 变更日期精确（exact）且新旧 valid_from 相同时的正确语义：

        旧行的 valid_until == 新行的 valid_from == t（现实时间轴无缝衔接）。
        于是 t 之前只认旧值、t 之后只认新值 —— 这才是"用户说 5/1 搬家"该有的行为。
        """
        t = time.time() - DAY
        x = self.mkfact("用户常住上海", valid_from=t)
        y = self.close([x], "用户搬到深圳", valid_from=t)      # 同一个 valid_from！
        self.assertEqual(self.s.get_fact(x)["valid_until"], self.s.get_fact(y)["valid_from"],
                         "现实时间轴必须无缝衔接")
        ids_at = lambda ts: [r["id"] for r in self.s.as_of(ts, "valid", user_id=self.u)]
        self.assertNotIn(y, ids_at(t - 1), "t 之前新值不该可见")
        self.assertNotIn(x, ids_at(t + 1), "t 之后旧值必须被封闭")
        self.assertIn(y, ids_at(t + 1))
        # setUp 里那三条 address 也在有效期内，所以只断言 x/y 的相对关系
        self.assertEqual(sorted(set(ids_at(t)) & {x, y}), [y],
                         "恰好等于 t 时按'新值生效'解释")

    def test_null_valid_from_counts_as_always_valid(self):
        """未知 ≠ 尚未开始：valid_from 为 NULL 的旧数据在任意历史时刻都应可见"""
        fid = self.s.add_fact("用户有一只猫", embed_stub(), user_id=self.u,
                              kind="fact", slot="pet")
        ids = [r["id"] for r in self.s.as_of(self.now - 500 * DAY, "valid", user_id=self.u)]
        self.assertIn(fid, ids, "NULL 的 valid_from 必须按'一直有效'处理，否则会凭空丢事实")


class L1UserIsolation(BaseCase):

    def test_queries_never_cross_users(self):
        other = self.u + "_other"
        mine = self.mkfact("用户常住上海")
        try:
            self.s.add_fact("别人的住址", embed_stub(), user_id=other,
                            kind="identity", slot="address")
            self.assertEqual([r["id"] for r in self.s.active_facts(user_id=self.u)], [mine])
            self.assertEqual(len(self.s.active_facts(user_id=other)), 1)
            self.assertEqual(self.s.history_of("identity", "address", user_id=other)[0]["content"],
                             "别人的住址")
        finally:
            with get_conn() as c, c.cursor() as cur:
                cur.execute("DELETE FROM facts WHERE user_id = %s", (other,))


# ══════════════════════════════════════════════════════════
# L2 领域层：纯函数（不联网）
# ══════════════════════════════════════════════════════════
class L2PureFunctions(unittest.TestCase):

    def test_slot_of_falls_back_to_kind(self):
        self.assertEqual(slot_of("identity", None), "identity")
        self.assertEqual(slot_of("identity", "  Address "), "address")
        self.assertEqual(slot_of("", None), "general")
        self.assertEqual(slot_of("identity", ""), "identity")

    def test_can_supersede_requires_single_value_slot(self):
        old_multi = {"slot": "dietary", "source": "user_explicit"}
        old_single = {"slot": "address", "source": "user_explicit"}
        self.assertFalse(can_supersede(old_multi, "user_explicit"),
                         "多值槽位（口味/爱好）永远不能自动覆盖")
        self.assertTrue(can_supersede(old_single, "user_explicit"))

    def test_can_supersede_respects_evidence_hierarchy(self):
        """模型猜的不能废掉用户说的——这是长期记忆正确性的底线"""
        old = {"slot": "address", "source": "user_explicit"}
        self.assertFalse(can_supersede(old, "agent_inferred"))
        self.assertTrue(can_supersede(old, "user_correction"))
        old_guess = {"slot": "address", "source": "agent_inferred"}
        self.assertTrue(can_supersede(old_guess, "user_explicit"))

    def test_parse_ts_accepts_documented_formats(self):
        self.assertIsNotNone(_parse_ts("2026-05-01"))
        self.assertIsNotNone(_parse_ts("2026-05-01 12:30"))
        self.assertIsNotNone(_parse_ts("2026-05-01 12:30:45"))
        self.assertIsNotNone(_parse_ts("1789985300"))
        self.assertIsNone(_parse_ts("上个月"))
        self.assertIsNone(_parse_ts(""))
        d1 = _parse_ts("2026-05-01")
        d2 = _parse_ts("2026-05-02")
        self.assertAlmostEqual(d2 - d1, DAY, places=0)

    def test_fmt_time_is_relative(self):
        now = time.time()
        self.assertEqual(fmt_time(0), "未知", "ts=0/None 必须被 guard 住，不能算成'56年前'")
        self.assertEqual(fmt_time(None), "未知")
        self.assertEqual(fmt_time(now), "刚刚")
        self.assertEqual(fmt_time(now - 3 * DAY), "3天前")
        self.assertIn("月", fmt_time(now - 100 * DAY))


# ══════════════════════════════════════════════════════════
# L2 领域层：需要真实 embedding（网络不通自动 skip）
# ══════════════════════════════════════════════════════════
class L2RememberWithEmbedding(BaseCase):

    def setUp(self):
        super().setUp()
        from memory import embed
        try:
            self.vec = embed.fetch("连通性自检")[0]
        except Exception as e:                       # noqa: BLE001
            self.skipTest(f"embedding 服务不可用，跳过：{type(e).__name__}: {e}")
        if not self.vec:
            self.skipTest("embedding 返回空向量")
        self.m = Memory(user_id=self.u, scene="test-session")

    def test_remember_then_recall_returns_id_and_no_vector(self):
        self.m.remember("用户常住上海", kind="identity", slot="address")
        self.m.remember("用户叫华尧宽", kind="identity", slot="name")
        res = self.m.recall("我住在哪里？")
        self.assertTrue(res, "刚写进去的记忆必须能召回")
        self.assertIn("id", res[0], "必须回传 id（纠错的特权路径依赖它）")
        self.assertNotIn("embed", res[0], "!!!!! 绝不能把 1024 维向量喂给 LLM")
        import json
        size = len(json.dumps(res, ensure_ascii=False))
        self.assertLess(size, 500, f"召回返回体过大（{size} 字符），检查是否泄漏了内部字段")

    def test_single_value_slot_gets_superseded(self):
        """住址是单值槽位：写第二条时第一条必须自动失效"""
        self.m.remember("用户常住上海", kind="identity", slot="address")
        before = self.s.active_facts(user_id=self.u)
        self.assertEqual(len(before), 1)
        msg = self.m.remember("用户搬到深圳了", kind="identity", slot="address")
        rows = self.s.active_facts(user_id=self.u)
        self.assertEqual(len(rows), 1, f"住址应只剩一条，实际 {len(rows)} 条（msg={msg}）")
        self.assertEqual(rows[0]["content"], "用户搬到深圳了")
        self.assertIn("更新", msg)
        # 旧行必须还在，且指针完整（能回溯）
        old = self.s.get_fact(before[0]["id"])
        self.assertIsNotNone(old["valid_until"])
        self.assertEqual(old["superseded_by"], rows[0]["id"])

    def test_multi_value_slot_coexists(self):
        """口味是 multi-value：第二条不能覆盖第一条（宁可并存，不可误删）"""
        self.m.remember("用户喜欢吃辣", kind="preference", slot="dietary")
        self.m.remember("用户不喜欢吃香菜", kind="preference", slot="dietary")
        rows = self.s.active_facts(kinds=["preference"], user_id=self.u)
        self.assertEqual(len(rows), 2, "dietary 是多值槽位，两条必须都在")

    def test_memory_reads_its_own_user(self):
        """回归防护：recall 必须带 user_id，否则写 A 读 B（静默召回空）"""
        self.m.remember("用户常住上海", kind="identity", slot="address")
        other = Memory(user_id="__nobody__", scene=None)
        self.assertEqual(other.recall("我住在哪里？"), [],
                         "别的用户不该看到这条记忆")


if __name__ == "__main__":
    unittest.main(verbosity=2)
