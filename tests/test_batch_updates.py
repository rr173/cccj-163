"""批次增量更新的完整正确性套件。

核心策略：维护一份 *独立的内存全量重算器*（直接在 Python 袋上做修订与
嵌套循环连接，与增量引擎不共享任何代码），每个批次之后从交付文件重开
查询会话，逐项比较：

1. 总重数、全序每一页（含多个不同位置的小页）、条件筛选后的连续秩；
2. 每条输出元组的来源表达——组合数、source_refs、participates，
   与独立逐行枚举 brute_force_lineage 完全一致；
3. 撤销记录不再出现在新版本来源表达，未改记录仍可被原 row_seq 引用。

覆盖需求点名的全部场景：多角色共用同一关系、两端同时增加、同内容不同
身份、重复次数加减、撤销、投影后结果重合、null 键不匹配、先清空再恢复、
批次重放与编号冲突；另测提交前故障/强制退出的原子性与成功后的重开恢复。
"""
import os
import sys
import tempfile
import unittest
import subprocess
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from factor_delivery import (
    Column, Table, Edge, Spec,
    build_factorized, DeliveryStore, QuerySession, Condition,
    ChangeSet, BatchChange, BatchConflictError, apply_batch,
    brute_force_join, brute_force_lineage,
)


# --------------------------------------------------------------------- #
# 独立内存重算器：存活记录袋（rid -> (values, mult)），全量重算 Spec
# --------------------------------------------------------------------- #
class InMemoryState:
    def __init__(self, spec: Spec):
        self.tables = spec.tables
        # 每表：rid -> (row tuple, mult, row_seq)。
        # rid 与 row_seq 高水位都只增、撤销不复用，与交付文件物理语义一致。
        self.live: dict = {t: {} for t in spec.tables}
        self.next_rid = {t: 0 for t in spec.tables}
        self.next_seq = {t: 0 for t in spec.tables}
        for tname, tbl in spec.tables.items():
            for row, mult in zip(tbl.rows, tbl.multiplicities):
                rid = self.next_rid[tname]
                seq = self.next_seq[tname]
                self.live[tname][rid] = (tuple(row), mult, seq)
                self.next_rid[tname] += 1
                self.next_seq[tname] += 1
        self.spec_template = spec

    def insert(self, tname, values, mult):
        rid = self.next_rid[tname]
        seq = self.next_seq[tname]
        self.next_rid[tname] += 1
        self.next_seq[tname] += 1
        self.live[tname][rid] = (tuple(values), mult, seq)
        return rid

    def retract(self, tname, rid):
        del self.live[tname][rid]
        # rid / row_seq 高水位均不回退

    def update_mult(self, tname, rid, mult):
        values, _old, seq = self.live[tname][rid]
        self.live[tname][rid] = (values, mult, seq)

    def to_spec(self) -> Spec:
        t = self.spec_template
        tables = {}
        for tname, tbl in t.tables.items():
            # 按 row_seq 装载顺序排列（与物理表一致，撤销产生序号空洞）
            ordered = sorted(self.live[tname].items(),
                             key=lambda kv: kv[1][2])
            rows = [v for _rid, (v, _m, _s) in ordered]
            mults = [m for _rid, (_v, m, _s) in ordered]
            seqs = [s for _rid, (_v, _m, s) in ordered]
            tables[tname] = Table(
                [Column(c.name, c.dtype, c.nullable, c.attr) for c in tbl.columns],
                rows, mults, row_seqs=seqs,
            )
        return Spec(
            tables=tables,
            instances=dict(t.instances),
            edges=list(t.edges),
            output_attrs=list(t.output_attrs),
            name=t.name,
        )


# --------------------------------------------------------------------- #
# 全量比对工具
# --------------------------------------------------------------------- #
def expanded_bag(spec):
    out = []
    for tup, mult in brute_force_join(spec):
        out.extend([tup] * mult)
    return out


def assert_snapshot_equal(test, path, state: InMemoryState,
                          check_lineage=True):
    """重开交付文件，与独立全量重算结果全面比对。"""
    spec = state.to_spec()
    expected = brute_force_join(spec)
    expected_total = sum(m for _, m in expected)
    store = DeliveryStore.open(path)
    try:
        sess = QuerySession(store)
        test.assertEqual(sess.count(), expected_total)
        if expected_total:
            exp_rows = expanded_bag(spec)
            # 不同位置的小页：开头、1/4、中点、3/4、末尾
            spots = sorted({0, expected_total // 4, expected_total // 2,
                            (3 * expected_total) // 4, expected_total - 1})
            for r in spots:
                n = min(3, expected_total - r)
                test.assertEqual(sess.page(r, n), exp_rows[r:r + n])
            # 全量分页覆盖
            pages, r = [], 0
            while r < expected_total:
                chunk = sess.page(r, 7)
                test.assertTrue(chunk)
                pages.extend(chunk)
                r += len(chunk)
            test.assertEqual(pages, exp_rows)
            # 条件筛选：对首列若干取值做等值/is_null，重新编号核对
            if spec.output_attrs:
                attr = spec.output_attrs[0]
                vals = sorted({t[0] for t, _ in expected},
                              key=lambda v: spec.value_key(attr, v))
                sample = vals[:3] + vals[-2:]
                for v in sample:
                    conds = [Condition.is_null(attr)] if v is None \
                        else [Condition.eq(attr, v)]
                    view = sess.restricted(conds)

                    def matches(t):
                        return (t[0] is None and v is None) or \
                               (t[0] is not None and t[0] == v)

                    exp_count = sum(m for t, m in expected if matches(t))
                    test.assertEqual(view.count(), exp_count)
                    if exp_count:
                        exp_expanded = []
                        for tup, mult in expected:
                            if matches(tup):
                                exp_expanded.extend([tup] * mult)
                        test.assertEqual(view.page(0, exp_count), exp_expanded)
            if check_lineage:
                _assert_lineage_equal(test, sess, spec, state)
        else:
            test.assertEqual(sess.page(0, 5), [])
    finally:
        store.close()


def _assert_lineage_equal(test, sess, spec, state: InMemoryState):
    enum = brute_force_lineage(spec)
    # 每条输出元组：组合数、来源引用、参与判断全部独立核对
    for tup, combos in enum.items():
        expr = sess.lineage(tup)
        test.assertTrue(expr.exists(), f"元组 {tup} 应存在")
        expect_weight = sum(combos.values())
        test.assertEqual(expr.combination_count(), expect_weight,
                         f"{tup} 组合数不符")
        # source_refs：独立枚举里出现过的 (alias, row_seq, mult) 集合
        expect_refs = set()
        for combo, weight in combos.items():
            for alias, seq in combo:
                expect_refs.add((alias, seq))
        got_refs = {(a, s) for a, s, _m in expr.source_refs()}
        test.assertEqual(got_refs, expect_refs,
                         f"{tup} 来源引用不符: {got_refs} vs {expect_refs}")
        # participates：参与序号必须命中；未参与/已撤销序号必须为 False。
        # row_seq 高水位可能因撤销产生空洞，按物理表高水位遍历。
        for alias in spec.instances:
            participating = {s for (a, s) in expect_refs if a == alias}
            for seq in participating:
                test.assertTrue(
                    expr.participates(alias, seq),
                    f"{tup}: {alias}#{seq} 应参与",
                )
            tname = spec.instances[alias]
            phys = sess.store.src_physical(tname)
            (hw,) = sess.conn.execute(
                f"SELECT COALESCE(MAX(row_seq), -1) FROM {phys}"
            ).fetchone()
            for seq in range(hw + 1):
                if (alias, seq) not in expect_refs:
                    test.assertFalse(
                        expr.participates(alias, seq),
                        f"{tup}: {alias}#{seq} 不应参与",
                    )
    # 不存在的输出元组：按输出列声明类型构造一个域外值，类型合法但不存在
    if enum:
        bad = tuple(_domain_absent_value(spec, attr, enum)
                    for attr in spec.output_attrs)
        if bad not in enum and all(v is not None for v in bad):
            test.assertFalse(sess.lineage(bad).exists())


def _domain_absent_value(spec: Spec, attr: str, enum) -> Any:
    """选一个类型合法、但不在任何已存在元组该列位置上出现的值。"""
    dtype = spec.attr_type[attr]
    used = {t[spec.output_attrs.index(attr)] for t in enum}
    if dtype == "int64":
        v = 987654321
        while v in used:
            v += 1
        return v
    if dtype == "bool":
        return not all(used) if used else True
    v = "__absent__"
    while v in used:
        v = "_" + v
    return v


# --------------------------------------------------------------------- #
# 通用三表星型规格（device/left/right），可切投影与可空
# --------------------------------------------------------------------- #
def star_spec(devices, parts, supplies, output=None, nullable=True,
              mults=None, name="inc"):
    device = Table([Column("id", "int64", nullable, "device")],
                   [(d,) for d in devices])
    lm = mults["left"] if mults else None
    rm = mults["right"] if mults else None
    left = Table(
        [Column("dev", "int64", nullable, "device"),
         Column("part", "text", False, "part")],
        parts, lm,
    )
    right = Table(
        [Column("dev", "int64", nullable, "device"),
         Column("sup", "text", False, "supply")],
        supplies, rm,
    )
    return Spec(
        tables={"device": device, "left": left, "right": right},
        instances={"d": "device", "l": "left", "r": "right"},
        edges=[Edge("d", ("id",), "l", ("dev",)),
               Edge("d", ("id",), "r", ("dev",))],
        output_attrs=output or ["device", "part", "supply"],
        name=name,
    )


class TestBatchUpdates(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.path = os.path.join(self.tmp, "inc.db")

    def _build(self, spec):
        store = DeliveryStore.create(self.path, spec)
        build_factorized(store, spec)
        store.close()
        return InMemoryState(spec)

    def _apply(self, state, batch_no, changes, crash=None):
        store = DeliveryStore.open(self.path)
        try:
            cs = ChangeSet(batch_no)
            for ch in changes:
                cs.add(ch)
            result = apply_batch(store, cs, crash=crash)
            # 同步内存重算器（成功路径）。内存侧自己分配 rid，再用交付
            # 文件返回的 rid 核对身份生命周期（插入只增、撤销不复用）。
            if crash is None:
                for pos, ch in enumerate(changes):
                    if ch.op == "insert":
                        assigned_rid = result.inserted_rid[str(pos)][1]
                        local_rid = state.insert(ch.table, ch.values, ch.mult)
                        self.assertEqual(
                            assigned_rid, local_rid,
                            "系统分配 rid 必须与独立重算器的高水位身份一致",
                        )
                    elif ch.op == "retract":
                        state.retract(ch.table, ch.rid)
                    else:
                        state.update_mult(ch.table, ch.rid, ch.mult)
            return result
        finally:
            store.close()

    # 1. 两端同时增加：交互增量不能由两次单侧差额推得 ---------------- #
    def test_both_sides_insert_interactive_increment(self):
        spec = star_spec([1, 2],
                         [(1, "a"), (2, "c")],
                         [(1, "x"), (2, "q")])
        state = self._build(spec)
        assert_snapshot_equal(self, self.path, state)
        self._apply(state, "B1", [
            BatchChange.insert("left", (2, "e"), 1),
            BatchChange.insert("right", (2, "z"), 1),
        ])
        assert_snapshot_equal(self, self.path, state)

    # 2. 同内容不同身份：新记录不复用旧 rid/row_seq -------------------- #
    def test_same_content_distinct_identity(self):
        spec = star_spec([1], [(1, "a")], [(1, "x")])
        state = self._build(spec)
        r = self._apply(state, "B1", [
            BatchChange.insert("left", (1, "a"), 1),
        ])
        new_rid = r.inserted_rid["0"][1]
        self.assertNotEqual(new_rid, 0)
        # row_seq 高水位：新序号在旧序号之后，两行值相同但身份独立
        store = DeliveryStore.open(self.path)
        try:
            row0 = store.fetch_source_row("l", 0)
            row_new = store.fetch_source_row("l", 1)
            self.assertIsNotNone(row0)
            self.assertIsNotNone(row_new)
            self.assertEqual(row0["values"], row_new["values"])
            self.assertEqual(row0["row_seq"], 0)
            self.assertEqual(row_new["row_seq"], 1)
        finally:
            store.close()
        assert_snapshot_equal(self, self.path, state)
        # 撤销新身份，旧身份仍在
        self._apply(state, "B2", [BatchChange.retract("left", new_rid)])
        assert_snapshot_equal(self, self.path, state)

    # 3. 重复次数加减 ------------------------------------------------- #
    def test_multiplicity_increase_and_decrease(self):
        spec = star_spec([1], [(1, "a")], [(1, "x")],
                         mults={"left": [2], "right": [3]})
        state = self._build(spec)
        # 找到 left rid 0
        self._apply(state, "B1", [
            BatchChange.update_mult("left", 0, 5),
        ])
        assert_snapshot_equal(self, self.path, state)
        self._apply(state, "B2", [
            BatchChange.update_mult("left", 0, 1),
            BatchChange.update_mult("right", 0, 7),
        ])
        assert_snapshot_equal(self, self.path, state)
        # 改回 1，与无 mult 基线一致
        self._apply(state, "B3", [
            BatchChange.update_mult("right", 0, 1),
        ])
        assert_snapshot_equal(self, self.path, state)

    # 4. 撤销（含撤销后投影处抵消）------------------------------------ #
    def test_retract_records_disappear_from_lineage(self):
        spec = star_spec([1, 2],
                         [(1, "a"), (1, "b"), (2, "c")],
                         [(1, "x"), (2, "q")])
        state = self._build(spec)
        self._apply(state, "B1", [
            BatchChange.retract("left", 1),   # 撤销 (1,b)
            BatchChange.retract("right", 1),  # 撤销 (2,q)
        ])
        assert_snapshot_equal(self, self.path, state)
        # 被撤销 row_seq 不再可审计
        store = DeliveryStore.open(self.path)
        try:
            self.assertIsNone(store.fetch_source_row("l", 1))
        finally:
            store.close()

    # 5. 投影后结果重合：alt 抵消/叠加在投影合并处正确 ---------------- #
    def test_projection_overlap_union_merge(self):
        # 输出不含 device：不同设备产生相同 (part, supply) 时重数相加
        spec = star_spec(
            [1, 2],
            [(1, "a"), (2, "a")],
            [(1, "x"), (2, "x")],
            output=["part", "supply"],
        )
        state = self._build(spec)
        assert_snapshot_equal(self, self.path, state)
        # 再插一条让两侧设备投影后继续重合/抵消
        self._apply(state, "B1", [
            BatchChange.insert("left", (1, "a"), 1),
            BatchChange.insert("right", (2, "x"), 1),
        ])
        assert_snapshot_equal(self, self.path, state)
        self._apply(state, "B2", [
            BatchChange.retract("left", 0),
        ])
        assert_snapshot_equal(self, self.path, state)

    # 6. null 键不匹配 ------------------------------------------------ #
    def test_null_key_never_matches(self):
        spec = star_spec([1, 2],
                         [(1, "a"), (None, "n")],
                         [(1, "x"), (None, "m")])
        state = self._build(spec)
        assert_snapshot_equal(self, self.path, state)
        self._apply(state, "B1", [
            BatchChange.insert("left", (None, "n2"), 1),
            BatchChange.insert("right", (1, "y"), 1),
        ])
        assert_snapshot_equal(self, self.path, state)
        self._apply(state, "B2", [
            BatchChange.retract("left", 2),
            BatchChange.retract("right", 1),
        ])
        assert_snapshot_equal(self, self.path, state)

    # 7. 先清空再恢复 ------------------------------------------------- #
    def test_clear_then_restore(self):
        spec = star_spec([1, 2],
                         [(1, "a"), (2, "c")],
                         [(1, "x"), (2, "q")])
        state = self._build(spec)
        # 清空左右两表全部记录（分两个批次）
        self._apply(state, "B1", [
            BatchChange.retract("left", 0),
            BatchChange.retract("left", 1),
        ])
        assert_snapshot_equal(self, self.path, state, check_lineage=False)
        store = DeliveryStore.open(self.path)
        try:
            self.assertEqual(QuerySession(store).count(), 0)
        finally:
            store.close()
        self._apply(state, "B2", [
            BatchChange.retract("right", 0),
            BatchChange.retract("right", 1),
        ])
        assert_snapshot_equal(self, self.path, state, check_lineage=False)
        # 恢复：插入全新身份（rid 不复用），结果回到初始语义
        self._apply(state, "B3", [
            BatchChange.insert("left", (1, "a"), 1),
            BatchChange.insert("left", (2, "c"), 1),
            BatchChange.insert("right", (1, "x"), 1),
            BatchChange.insert("right", (2, "q"), 1),
        ])
        assert_snapshot_equal(self, self.path, state)

    # 8. 批次重放（同号同内容）与编号冲突（同号异内容）-------------- #
    def test_replay_and_conflict(self):
        spec = star_spec([1], [(1, "a")], [(1, "x")])
        state = self._build(spec)
        changes = [BatchChange.insert("left", (1, "b"), 1)]
        r1 = self._apply(state, "DUP", changes)
        # 重开后重放：返回原结果，不产生新版本
        store = DeliveryStore.open(self.path)
        try:
            r2 = apply_batch(store, ChangeSet("DUP", list(changes)))
            self.assertEqual(r1.applied_seq, r2.applied_seq)
            self.assertEqual(r1.total_weight, r2.total_weight)
            self.assertEqual(r1.content_hash, r2.content_hash)
            # 同号异内容拒绝
            with self.assertRaises(BatchConflictError):
                apply_batch(store, ChangeSet("DUP", [
                    BatchChange.insert("left", (1, "zzz"), 1),
                ]))
        finally:
            store.close()
        assert_snapshot_equal(self, self.path, state)

    # 9. 多角色共用同一物理关系：一次修订所有角色同时看到 ------------- #
    def test_same_table_multiple_roles_one_revision(self):
        # 沿用产品既有的同表多实例合法模式：node 表被 parent / child 两个
        # 角色复用（共享逻辑属性 node_id 即连接键），edge 表承载连接与载荷。
        # 对 node 的一次物理修订必须同时反映到两个角色，且只提交一次。
        node_tbl = Table(
            [Column("id", "int64", False, "node_id")],
            [(1,), (2,)],
        )
        edge_tbl = Table(
            [Column("p", "int64", False, "node_id"),
             Column("c", "int64", False, "node_id"),
             Column("pl", "text", False, "parent_label"),
             Column("cl", "text", False, "child_label")],
            [(1, 2, "root", "leaf")],
        )
        spec = Spec(
            tables={"node": node_tbl, "edge": edge_tbl},
            instances={"parent": "node", "child": "node", "e": "edge"},
            edges=[Edge("parent", ("id",), "e", ("p",)),
                   Edge("child", ("id",), "e", ("c",))],
            output_attrs=["node_id", "parent_label", "child_label"],
        )
        state = self._build(spec)
        assert_snapshot_equal(self, self.path, state)
        # 物理表 node 只提交一次修订，parent/child 两个角色同时看到
        self._apply(state, "B1", [
            BatchChange.insert("node", (3,), 1),
        ])
        assert_snapshot_equal(self, self.path, state)
        # 撤销孤立新 rid（id=3，无任何 edge 引用）：两个角色的因子都要
        # 同步失去该分区，但既有连接结果不受影响。
        new_rid = 2  # B1 插入的 node 记录
        self._apply(state, "B2", [
            BatchChange.retract("node", new_rid),
        ])
        assert_snapshot_equal(self, self.path, state)

    # 10. 输入校验失败不留半成品 -------------------------------------- #
    def test_validation_failure_rolls_back(self):
        spec = star_spec([1], [(1, "a")], [(1, "x")])
        state = self._build(spec)
        before = DeliveryStore.open(self.path)
        try:
            before_count = QuerySession(before).count()
            before_batches = len(before.list_batches())
        finally:
            before.close()
        store = DeliveryStore.open(self.path)
        try:
            with self.assertRaises(Exception):
                apply_batch(store, ChangeSet("BAD", [
                    BatchChange.retract("left", 999),  # 不存在的 rid
                ]))
            with self.assertRaises(Exception):
                apply_batch(store, ChangeSet("BAD2", [
                    BatchChange.insert("left", (1, 123), 1),  # 类型错误
                ]))
        finally:
            store.close()
        store = DeliveryStore.open(self.path)
        try:
            self.assertEqual(QuerySession(store).count(), before_count)
            self.assertEqual(len(store.list_batches()), before_batches)
        finally:
            store.close()
        assert_snapshot_equal(self, self.path, state)

    # 11. 提交前异常：回滚到旧版本 ------------------------------------ #
    def test_crash_before_commit_keeps_old_version(self):
        spec = star_spec([1], [(1, "a")], [(1, "x")])
        state = self._build(spec)
        store = DeliveryStore.open(self.path)
        try:
            old_count = QuerySession(store).count()
            with self.assertRaises(RuntimeError):
                apply_batch(store, ChangeSet("CRASH", [
                    BatchChange.insert("left", (1, "b"), 1),
                ]), crash="before_commit")
        finally:
            store.close()
        # 重开：旧版本完整，批次未登记
        store = DeliveryStore.open(self.path)
        try:
            sess = QuerySession(store)
            self.assertEqual(sess.count(), old_count)
            self.assertEqual(
                [b["batch_no"] for b in store.list_batches()], []
            )
        finally:
            store.close()

    # 12. 强制退出：未提交事务随进程消失，重开只读旧版本 -------------- #
    def test_hard_exit_before_commit_keeps_old_version(self):
        spec = star_spec([1], [(1, "a")], [(1, "x")])
        self._build(spec)
        runner = os.path.join(self.tmp, "hard_exit_runner.py")
        with open(runner, "w", encoding="utf-8") as f:
            f.write(_HARD_EXIT_RUNNER.replace("__DB__", self.path))
        proc = subprocess.run(
            [sys.executable, "-I", runner],
            capture_output=True, text=True,
        )
        self.assertEqual(proc.returncode, 7, proc.stderr)
        store = DeliveryStore.open(self.path)
        try:
            sess = QuerySession(store)
            self.assertEqual(sess.count(), 1)  # 初始 1*1*1
            self.assertEqual(
                [b["batch_no"] for b in store.list_batches()], []
            )
            # 同一编号在重开后仍可正常使用（未占用）
            result = apply_batch(store, ChangeSet("HX", [
                BatchChange.insert("left", (1, "b"), 1),
            ]))
            self.assertEqual(result.applied_seq, 1)
        finally:
            store.close()

    # 13. 成功提交后新进程恢复最新版本与批次去重信息 ------------------ #
    def test_reopen_restores_version_and_dedup(self):
        spec = star_spec([1], [(1, "a")], [(1, "x")])
        state = self._build(spec)
        self._apply(state, "OK1", [
            BatchChange.insert("left", (1, "b"), 1),
        ])
        self._apply(state, "OK2", [
            BatchChange.insert("right", (1, "y"), 1),
        ])
        store = DeliveryStore.open(self.path)
        try:
            batches = store.list_batches()
            self.assertEqual([b["batch_no"] for b in batches],
                             ["OK1", "OK2"])
            self.assertEqual([b["applied_seq"] for b in batches], [1, 2])
            # 重放 OK1 仍返回 seq=1 的原结果
            r = apply_batch(store, ChangeSet("OK1", [
                BatchChange.insert("left", (1, "b"), 1),
            ]))
            self.assertEqual(r.applied_seq, 1)
        finally:
            store.close()
        assert_snapshot_equal(self, self.path, state)

    # 14. 随机多批次混合操作：每批次后都与独立全量重算逐项一致 -------- #
    def test_random_mixed_batches_match_full_recompute(self):
        import random
        rng = random.Random(20261117)
        devices = [1, 2, 3]
        spec = star_spec(
            devices,
            [(1, "a"), (2, "b"), (3, "c")],
            [(1, "x"), (2, "y"), (3, "z")],
        )
        state = self._build(spec)
        assert_snapshot_equal(self, self.path, state)
        seq = 0
        for trial in range(24):
            changes = []
            # 每个批次随机混合 1~3 条修订
            for _ in range(rng.randint(1, 3)):
                tname = rng.choice(["left", "right"])
                op = rng.choice(["insert", "insert", "retract", "update"])
                live_rids = sorted(state.live[tname].keys())
                if op in ("retract", "update") and live_rids:
                    rid = rng.choice(live_rids)
                    if op == "retract":
                        changes.append(BatchChange.retract(tname, rid))
                    else:
                        changes.append(
                            BatchChange.update_mult(tname, rid,
                                                    rng.randint(1, 4)))
                else:
                    dev = rng.choice(devices + [None, 99])  # 含 null/孤立键
                    if tname == "left":
                        val = (dev, rng.choice(["a", "b", "c", "d"]))
                    else:
                        val = (dev, rng.choice(["x", "y", "z", "w"]))
                    changes.append(
                        BatchChange.insert(tname, val, rng.randint(1, 3)))
            seq += 1
            # 跳过会因 rid 已撤销而校验失败的批次（语义上本就该拒绝），
            # 用独立 try 记录：被引擎拒绝的批次，内存侧也不应用。
            try:
                self._apply(state, f"R{seq:02d}", changes)
            except Exception:
                # 引擎拒绝（如对同 rid 重复修订）属预期，内存侧未改，
                # 交付文件也应保持上一快照不变。
                assert_snapshot_equal(self, self.path, state,
                                      check_lineage=False)
            # 每若干批次做一次完整快照核对（含来源）
            if trial % 4 == 3:
                assert_snapshot_equal(self, self.path, state)
        assert_snapshot_equal(self, self.path, state)


_HARD_EXIT_RUNNER = '''
import sys
sys.path.insert(0, "/workspace")
from factor_delivery import (
    DeliveryStore, ChangeSet, BatchChange, apply_batch,
)
store = DeliveryStore.open("__DB__")
apply_batch(store, ChangeSet("HX", [
    BatchChange.insert("left", (1, "b"), 1),
]), crash="hard_exit_before_commit")
raise AssertionError("不应到达这里")
'''


if __name__ == "__main__":
    unittest.main(verbosity=2)
