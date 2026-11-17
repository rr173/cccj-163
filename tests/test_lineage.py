"""来源追溯（lineage）正确性测试。

核对策略（两套完全独立的实现互相比对）：

1. reference.brute_force_lineage：逐行嵌套循环枚举 *每一个来源组合*，
   直接给出每条输出元组的 {来源组合: 权重}（小材料专用）；
2. factor_delivery.lineage 返回的紧凑表达：用本文件自带的独立展开器
   expand_expression 递归解释 src/prod/alt 节点，得到同样的组合袋。

两者必须逐项一致；同时专门覆盖需求点名的情形：
* 同一源行的 multiplicity（不逐份建节点，权重随行保留）；
* 值完全相同但装载序号不同的源行（仍是不同来源行）；
* 同一来源表被多个实例引用（角色身份分别保留）；
* 投影后不同连接分支得到同一输出元组（alt 相加，不交叉相乘）；
* 多个键分组中来源不可交叉组合；
* 重开文件（含新进程）后仍可追溯；不存在元组为空；类型/列错误明确报错。
"""
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from factor_delivery import (
    Column, Table, Edge, Spec,
    build_factorized, DeliveryStore, QuerySession, QueryError,
    brute_force_join, brute_force_lineage, lineage_participating_rows,
)
from factor_delivery.lineage import LineageExpression


# --------------------------------------------------------------------- #
# 独立表达展开器：只依赖 LineageNode 的公共字段，不调用被测导航逻辑
# --------------------------------------------------------------------- #
def expand_expression(expr: LineageExpression):
    """把来源表达解释成 {组合键(按别名排序的 (alias,row_seq) 元组): 权重}。

    src：行之间互斥（alt 语义）；prod：各分支组合笛卡尔积、权重相乘；
    alt：分支互斥、权重相加。multiplicity 在 src 行上作为权重参与，
    不逐份展开。
    """
    if not expr.exists():
        return {}

    def walk(node):
        if node.kind == "src":
            out = {}
            for seq, mult in node.rows:
                out[((node.alias, seq),)] = mult
            return out
        child_bags = [walk(c) for c in node.children]
        if node.kind == "alt":
            merged: dict = {}
            for bag in child_bags:
                for k, w in bag.items():
                    merged[k] = merged.get(k, 0) + w
            return merged
        # prod：逐分支笛卡尔积（组合键按实例别名排序的元组；嵌套 product
        # 展平后同一别名只出现一次，用 dict 去重）
        acc = {(): 1}
        for bag in child_bags:
            nxt = {}
            for k1, w1 in acc.items():
                for k2, w2 in bag.items():
                    merged = dict(k1)
                    merged.update(k2)
                    key = tuple(sorted(merged.items()))
                    nxt[key] = nxt.get(key, 0) + w1 * w2
            acc = nxt
        return acc

    return walk(expr.root)


def build(spec):
    path = os.path.join(tempfile.mkdtemp(), "lineage.db")
    store = DeliveryStore.create(path, spec)
    build_factorized(store, spec)
    store.close()
    store = DeliveryStore.open(path)
    return QuerySession(store), store, path


class LineageCheckMixin:
    def assert_lineage_matches_enumerator(self, sess, spec):
        enum = brute_force_lineage(spec)
        bag = brute_force_join(spec)
        # 1. 枚举器自身的组合权重之和 == 元组重数
        for tup, mult in bag:
            self.assertEqual(sum(enum[tup].values()), mult)
        # 2. 每条存在的输出元组：表达展开 == 枚举组合袋
        for tup, mult in bag:
            expr = sess.lineage(tup)
            self.assertTrue(expr.exists(), msg=f"{tup} 应存在")
            self.assertEqual(expr.combination_count(), mult)
            expanded = expand_expression(expr)
            self.assertEqual(expanded, enum[tup],
                             msg=f"来源组合不一致: {tup}")
            # 表达节点数有界：不超过来源行数 * 实例数 + 因子结构规模
            bound = sum(
                len(spec.tables[t].rows)
                for t in set(spec.instances.values())
            ) * (len(spec.instances) + 2) + 200
            self.assertLessEqual(expr.node_count(), bound)
            # 3. 参与判断：对每个实例的每个序号位逐一比对
            for alias in spec.instances:
                nrows = len(spec.tables[spec.instances[alias]].rows)
                expect = lineage_participating_rows(enum, tup, alias)
                for seq in range(nrows):
                    got = expr.participates(alias, seq)
                    self.assertEqual(got, seq in expect,
                                     msg=f"{tup} {alias}#{seq} 参与判断错误")
                # 越界序号不参与、不报错
                self.assertFalse(expr.participates(alias, nrows))
                self.assertFalse(expr.participates(alias, nrows + 5))
        # 4. 不存在的元组 -> 空表达
        absent = self._find_absent_tuple(spec, enum)
        if absent is not None:
            expr = sess.lineage(absent)
            self.assertFalse(expr.exists())
            self.assertEqual(expr.combination_count(), 0)
            self.assertEqual(expr.nodes(), [])
            self.assertEqual(expr.source_refs(), [])
        return enum

    def _find_absent_tuple(self, spec, enum):
        # 在输出列空间内挑一个确定不存在的元组
        present = set(enum)
        if not present:
            return tuple(None if spec.attr_nullable[a] else _dummy(spec, a)
                         for a in spec.output_attrs)
        sample = list(present)[0]
        for delta in range(-50, 51):
            cand = list(sample)
            for i, attr in enumerate(spec.output_attrs):
                if spec.attr_type[attr] == "int64":
                    cand[i] = (cand[i] or 0) + delta if delta else cand[i]
            t = tuple(cand)
            if t not in present and all(
                v is not None for v in t
            ):
                return t
        return None


def _dummy(spec, attr):
    return 0 if spec.attr_type[attr] == "int64" else (
        "z" if spec.attr_type[attr] == "text" else False)


class TestLineageSmallCases(unittest.TestCase, LineageCheckMixin):

    def test_multiplicity_kept_on_source_row(self):
        # 同一源行 multiplicity=4，表达里只能有一个 src 行节点，组合数含 4
        device = Table([Column("id", "int64", False, "device")],
                       [(1,)], multiplicities=[2])
        parts = Table(
            [Column("dev", "int64", False, "device"),
             Column("part", "text", False, "part")],
            [(1, "a")], multiplicities=[4],
        )
        spec = Spec(
            tables={"device": device, "parts": parts},
            instances={"d": "device", "l": "parts"},
            edges=[Edge("d", ("id",), "l", ("dev",))],
            output_attrs=["device", "part"],
        )
        sess, store, _ = build(spec)
        enum = self.assert_lineage_matches_enumerator(sess, spec)
        expr = sess.lineage((1, "a"))
        self.assertEqual(expr.combination_count(), 8)  # 2 * 4
        refs = {(a, s): m for a, s, m in expr.source_refs()}
        self.assertEqual(refs[("l", 0)], 4)  # 只有一行，mult 随行保留
        self.assertEqual(len([n for n in expr.nodes() if n.kind == "src"]), 2)
        # 组合袋里每个组合只出现一次（mult 不逐份展开）
        self.assertEqual(len(expand_expression(expr)), 1)
        self.assertEqual(enum[(1, "a")][(("d", 0), ("l", 0))], 8)
        store.close()

    def test_identical_rows_distinct_seqs_are_distinct_sources(self):
        # 值完全相同、序号不同的两行必须分别保留为两个来源行
        device = Table([Column("id", "int64", False, "device")], [(1,)])
        parts = Table(
            [Column("dev", "int64", False, "device"),
             Column("part", "text", False, "part")],
            [(1, "a"), (1, "a")],  # 完全相同的两行
            multiplicities=[1, 1],
        )
        spec = Spec(
            tables={"device": device, "parts": parts},
            instances={"d": "device", "l": "parts"},
            edges=[Edge("d", ("id",), "l", ("dev",))],
            output_attrs=["device", "part"],
        )
        sess, store, _ = build(spec)
        self.assert_lineage_matches_enumerator(sess, spec)
        expr = sess.lineage((1, "a"))
        refs = expr.source_refs()
        l_rows = sorted(s for a, s, _ in refs if a == "l")
        self.assertEqual(l_rows, [0, 1])  # 两个序号都在
        self.assertTrue(expr.participates("l", 0))
        self.assertTrue(expr.participates("l", 1))
        # 两个互斥来源组合，各权重 1
        expanded = expand_expression(expr)
        self.assertEqual(len(expanded), 2)
        self.assertEqual(sum(expanded.values()), 2)
        store.close()

    def test_same_source_table_two_instances_keep_identity(self):
        # node 表被 parent / child 两个角色复用：两实例来源身份独立
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
        sess, store, _ = build(spec)
        self.assert_lineage_matches_enumerator(sess, spec)
        # node_id 经连通属性绑定；与独立枚举器一致：两角色各取 node 行 #1
        expr = sess.lineage((2, "root", "leaf"))
        aliases = {n.alias for n in expr.nodes() if n.kind == "src"}
        self.assertEqual(aliases, {"parent", "child", "e"})
        # 两个角色身份独立保留，即使引用同一张来源表的同一序号
        self.assertTrue(expr.participates("parent", 1))
        self.assertFalse(expr.participates("parent", 0))
        self.assertTrue(expr.participates("child", 1))
        self.assertFalse(expr.participates("child", 0))
        # 可在 store 直接按实例 + 序号审计来源行（两角色各自独立查到同值）
        self.assertEqual(sess.source_row("parent", 1)["values"], {"id": 2})
        self.assertEqual(sess.source_row("child", 1)["values"], {"id": 2})
        self.assertIsNone(sess.source_row("child", 99))
        store.close()

    def test_projection_alt_branches_no_cross_product(self):
        # 投影掉 device：两个设备都产生 part=a，来源必须 alt 相加，
        # 不能出现 d1 部件行与 d2 设备行交叉的组合
        device = Table([Column("id", "int64", False, "device")],
                       [(1,), (2,)])
        parts = Table(
            [Column("dev", "int64", False, "device"),
             Column("part", "text", False, "part")],
            [(1, "a"), (2, "a"), (2, "b")],
        )
        spec = Spec(
            tables={"device": device, "parts": parts},
            instances={"d": "device", "l": "parts"},
            edges=[Edge("d", ("id",), "l", ("dev",))],
            output_attrs=["part"],
        )
        sess, store, _ = build(spec)
        enum = self.assert_lineage_matches_enumerator(sess, spec)
        expr = sess.lineage(("a",))
        self.assertEqual(expr.combination_count(), 2)
        combos = expand_expression(expr)
        # 恰好两个组合：(d#0,l#0) 与 (d#1,l#1)，没有交叉组合
        self.assertEqual(set(combos), {
            (("d", 0), ("l", 0)),
            (("d", 1), ("l", 1)),
        })
        self.assertEqual(enum[("a",)][(("d", 0), ("l", 0))], 1)
        self.assertEqual(enum[("a",)][(("d", 1), ("l", 1))], 1)
        store.close()

    def test_multiple_key_groups_do_not_cross_combine(self):
        # 三表星型：左/右表都按 device 分组，d1 的部件绝不能与 d2 的供应点组合
        device = Table([Column("id", "int64", False, "device")],
                       [(1,), (2,)])
        parts = Table(
            [Column("dev", "int64", False, "device"),
             Column("part", "text", False, "part")],
            [(1, "a"), (1, "b"), (2, "c")],
        )
        supply = Table(
            [Column("dev", "int64", False, "device"),
             Column("sup", "text", False, "supply")],
            [(1, "x"), (2, "q")],
        )
        spec = Spec(
            tables={"device": device, "parts": parts, "supply": supply},
            instances={"d": "device", "l": "parts", "r": "supply"},
            edges=[Edge("d", ("id",), "l", ("dev",)),
                   Edge("d", ("id",), "r", ("dev",))],
            output_attrs=["device", "part", "supply"],
        )
        sess, store, _ = build(spec)
        enum = self.assert_lineage_matches_enumerator(sess, spec)
        # (1,a,x) 的组合里 r 必须是 r#0(sup=x)，不能混入 r#1(sup=q)
        expr = sess.lineage((1, "a", "x"))
        combos = expand_expression(expr)
        for key in combos:
            pairs = dict(key)
            self.assertEqual(pairs["d"], 0)
            self.assertEqual(pairs["l"], 0)
            self.assertEqual(pairs["r"], 0)
        # (2,c,q) 对称
        expr2 = sess.lineage((2, "c", "q"))
        for key in expand_expression(expr2):
            pairs = dict(key)
            self.assertEqual(pairs["d"], 1)
            self.assertEqual(pairs["l"], 2)
            self.assertEqual(pairs["r"], 1)
        store.close()

    def test_chain_three_levels_lineage(self):
        a = Table([Column("ka", "int64", False, "ka"),
                   Column("av", "int64", False, "av")],
                  [(1, 10), (1, 11), (2, 12)])
        b = Table([Column("ka", "int64", False, "ka"),
                   Column("kb", "int64", False, "kb"),
                   Column("bv", "int64", False, "bv")],
                  [(1, 100, 1000), (2, 200, 2000)])
        c = Table([Column("kb", "int64", False, "kb"),
                   Column("cv", "int64", False, "cv")],
                  [(100, 10000), (200, 20000), (200, 20001)])
        spec = Spec(
            tables={"a": a, "b": b, "c": c},
            instances={"a": "a", "b": "b", "c": "c"},
            edges=[Edge("a", ("ka",), "b", ("ka",)),
                   Edge("b", ("kb",), "c", ("kb",))],
            output_attrs=["ka", "av", "kb", "bv", "cv"],
        )
        sess, store, _ = build(spec)
        self.assert_lineage_matches_enumerator(sess, spec)
        store.close()

    def test_composite_key_groups_isolated(self):
        # 多列键 (k1,k2)：不同键组的来源不能交叉
        a = Table(
            [Column("k1", "int64", False, "k1"),
             Column("k2", "text", False, "k2"),
             Column("av", "int64", False, "av")],
            [(1, "m", 10), (1, "n", 11), (2, "m", 12)],
        )
        b = Table(
            [Column("k1", "int64", False, "k1"),
             Column("k2", "text", False, "k2"),
             Column("bv", "int64", False, "bv")],
            [(1, "m", 20), (1, "n", 21), (2, "m", 22), (2, "n", 23)],
        )
        spec = Spec(
            tables={"a": a, "b": b},
            instances={"a": "a", "b": "b"},
            edges=[Edge("a", ("k1", "k2"), "b", ("k1", "k2"))],
            output_attrs=["k1", "k2", "av", "bv"],
        )
        sess, store, _ = build(spec)
        self.assert_lineage_matches_enumerator(sess, spec)
        store.close()

    def test_null_join_rows_never_in_lineage(self):
        a = Table(
            [Column("k", "int64", True, "k"),
             Column("av", "text", True, "av")],
            [(1, "p"), (None, "n1")],
        )
        b = Table(
            [Column("k", "int64", True, "k"),
             Column("bv", "text", True, "bv")],
            [(1, "x"), (None, "n2")],
        )
        spec = Spec(
            tables={"a": a, "b": b},
            instances={"a": "a", "b": "b"},
            edges=[Edge("a", ("k",), "b", ("k",))],
            output_attrs=["k", "av", "bv"],
        )
        sess, store, _ = build(spec)
        self.assert_lineage_matches_enumerator(sess, spec)
        expr = sess.lineage((1, "p", "x"))
        # null 键行 a#1 / b#1 被半连接约减，绝不参与任何结果
        self.assertFalse(expr.participates("a", 1))
        self.assertFalse(expr.participates("b", 1))
        self.assertTrue(expr.participates("a", 0))
        store.close()

    def test_empty_result_lineage_is_empty(self):
        device = Table([Column("id", "int64", False, "device")], [(1,)])
        left = Table(
            [Column("dev", "int64", False, "device"),
             Column("part", "text", False, "part")],
            [(9, "orphan")],
        )
        spec = Spec(
            tables={"device": device, "left": left},
            instances={"d": "device", "l": "left"},
            edges=[Edge("d", ("id",), "l", ("dev",))],
            output_attrs=["device", "part"],
        )
        sess, store, _ = build(spec)
        self.assertEqual(sess.count(), 0)
        expr = sess.lineage((1, "orphan"))
        self.assertFalse(expr.exists())
        self.assertEqual(expr.combination_count(), 0)
        store.close()

    def test_projection_overlap_with_multiplicity(self):
        # 投影重叠 + multiplicity：alt 各分支权重正确相加
        device = Table(
            [Column("id", "text", False, "device")],
            [("d",), ("d2",)], multiplicities=[2, 3],
        )
        left = Table(
            [Column("dev", "text", False, "device"),
             Column("part", "int64", False, "part")],
            [("d", 1), ("d2", 1)],
            multiplicities=[5, 7],
        )
        spec = Spec(
            tables={"device": device, "left": left},
            instances={"d": "device", "l": "left"},
            edges=[Edge("d", ("id",), "l", ("dev",))],
            output_attrs=["part"],
        )
        sess, store, _ = build(spec)
        enum = self.assert_lineage_matches_enumerator(sess, spec)
        expr = sess.lineage((1,))
        # 分支1: 2*5=10；分支2: 3*7=21；alt 相加 31，无交叉组合
        self.assertEqual(expr.combination_count(), 31)
        combos = expand_expression(expr)
        self.assertEqual(len(combos), 2)
        self.assertEqual(combos[(("d", 0), ("l", 0))], 10)
        self.assertEqual(combos[(("d", 1), ("l", 1))], 21)
        store.close()


class TestLineageErrors(unittest.TestCase):
    def _spec(self):
        device = Table([Column("id", "int64", True, "device")],
                       [(1,), (2,), (None,)])
        parts = Table(
            [Column("dev", "int64", True, "device"),
             Column("part", "text", True, "part")],
            [(1, "a"), (2, "a")],
        )
        supply = Table(
            [Column("dev", "int64", False, "device"),
             Column("sup", "bool", False, "sup")],
            [(1, True), (2, False)],
        )
        return Spec(
            tables={"device": device, "parts": parts, "supply": supply},
            instances={"d": "device", "l": "parts", "r": "supply"},
            edges=[Edge("d", ("id",), "l", ("dev",)),
                   Edge("d", ("id",), "r", ("dev",))],
            output_attrs=["device", "part", "sup"],
        )

    def test_type_and_arity_errors(self):
        sess, store, _ = build(self._spec())
        cases = [
            ("x", "a", True),     # int64 列传 str
            (1, 9, True),         # text 列传 int
            (1, "a", 1),          # bool 列传 int
            (1, "a"),             # 列数不足
            (1, "a", True, 1),    # 列数过多
            (1, "a", None),       # 不可空列传 NULL
        ]
        for bad in cases:
            with self.assertRaises(QueryError):
                sess.lineage(bad)
        # bool 列严格区分：True 合法，1 不合法
        self.assertTrue(sess.lineage((1, "a", True)).exists())
        # 可空列接受 NULL（作为取值参与匹配，可能落到空表达）
        expr = sess.lineage((None, "a", True))
        self.assertFalse(expr.exists())  # null 连接键不匹配
        # 非 tuple/list
        with self.assertRaises(QueryError):
            sess.lineage({1, "a", True})
        store.close()

    def test_participates_bad_alias_or_seq(self):
        sess, store, _ = build(self._spec())
        expr = sess.lineage((1, "a", True))
        with self.assertRaises(QueryError):
            expr.participates("nope", 0)
        with self.assertRaises(QueryError):
            expr.participates("d", -1)
        with self.assertRaises(QueryError):
            expr.participates("d", True)
        store.close()


class TestLineageReopen(unittest.TestCase):
    def test_reopen_in_new_process(self):
        device = Table([Column("id", "int64", False, "device")],
                       [(1,), (2,)])
        parts = Table(
            [Column("dev", "int64", False, "device"),
             Column("part", "text", False, "part")],
            [(1, "a"), (2, "a")],
        )
        supply = Table(
            [Column("dev", "int64", False, "device"),
             Column("sup", "text", False, "supply")],
            [(1, "x"), (2, "q")],
        )
        spec = Spec(
            tables={"device": device, "parts": parts, "supply": supply},
            instances={"d": "device", "l": "parts", "r": "supply"},
            edges=[Edge("d", ("id",), "l", ("dev",)),
                   Edge("d", ("id",), "r", ("dev",))],
            output_attrs=["device", "part", "supply"],
        )
        path = os.path.join(tempfile.mkdtemp(), "reopen.db")
        store = DeliveryStore.create(path, spec)
        build_factorized(store, spec)
        store.close()

        # 新进程：只有交付文件，没有任何内存 Spec / 输入文件
        code = (
            "import sys; sys.path.insert(0, %r);"
            "from factor_delivery import DeliveryStore, QuerySession;"
            "s = DeliveryStore.open(%r); q = QuerySession(s);"
            "e = q.lineage((1, 'a', 'x'));"
            "assert e.exists() and e.combination_count() == 1;"
            "assert e.participates('l', 0) and not e.participates('l', 1);"
            "assert q.lineage((2, 'a', 'x')).exists() is False;"
            "print('CHILD_OK', e.combination_count(), e.node_count())"
        ) % (os.path.dirname(os.path.dirname(os.path.abspath(__file__))), path)
        import subprocess
        proc = subprocess.run(
            [sys.executable, "-I", "-c", code],
            capture_output=True, text=True,
        )
        self.assertEqual(proc.returncode, 0, msg=proc.stderr)
        self.assertIn("CHILD_OK", proc.stdout)


if __name__ == "__main__":
    unittest.main(verbosity=2)
