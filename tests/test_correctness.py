"""小材料全量核对：因子化结果必须与独立逐行枚举的完整袋逐项一致。

覆盖：多列分隔键、源 multiplicity、null 连接不成立、空分支、
表角色/自身来源复用、投影去键后重数相加、bool/int/text 全序、条件重编号、
运行交集校验、不兼容输出顺序拒绝。
"""
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from factor_delivery import (
    Column, Table, Edge, Spec, SpecError,
    build_factorized, DeliveryStore, QuerySession, Condition,
    brute_force_join, brute_force_count, brute_force_filtered,
)
from factor_delivery.builder import BuildError


def build_session(spec):
    tmp = tempfile.mkdtemp()
    path = os.path.join(tmp, "case.db")
    store = DeliveryStore.create(path, spec)
    build_factorized(store, spec)
    store.close()
    store = DeliveryStore.open(path)
    return QuerySession(store), store


def expanded_bag(spec):
    out = []
    for tup, mult in brute_force_join(spec):
        out.extend([tup] * mult)
    return out


class CorrectnessMixin:
    def assert_matches_brute_force(self, spec):
        sess, store = build_session(spec)
        try:
            bag = brute_force_join(spec)
            expected_total = sum(m for _, m in bag)
            self.assertEqual(sess.count(), expected_total)

            # 每个秩逐行核对
            expanded = expanded_bag(spec)
            if expected_total:
                for r in (0, 1, expected_total // 2, expected_total - 1):
                    if r < expected_total:
                        self.assertEqual(sess.row_at(r), expanded[r])
                # 分页覆盖全部秩
                pages = []
                r = 0
                while r < expected_total:
                    chunk = sess.page(r, 3)
                    self.assertTrue(chunk)
                    pages.extend(chunk)
                    r += len(chunk)
                self.assertEqual(pages, expanded)
            # 越界与 n=0
            self.assertEqual(sess.page(expected_total, 5), [])
            self.assertEqual(sess.page(expected_total + 10, 5), [])
            self.assertEqual(sess.page(0, 0), [])
            return sess, store, bag
        finally:
            pass


class TestSmallCases(unittest.TestCase, CorrectnessMixin):

    def test_three_table_star_basic(self):
        device = Table([Column("id", "int64", False, "device")],
                       [(1,), (2,)])
        parts = Table(
            [Column("dev", "int64", False, "device"),
             Column("part", "text", False, "part")],
            [(1, "a"), (1, "b"), (2, "c"), (3, "z")],
        )
        supply = Table(
            [Column("dev", "int64", True, "device"),
             Column("sup", "text", False, "supply")],
            [(1, "x"), (1, "y"), (2, "q"), (None, "n")],
        )
        spec = Spec(
            tables={"device": device, "parts": parts, "supply": supply},
            instances={"d": "device", "l": "parts", "r": "supply"},
            edges=[Edge("d", ("id",), "l", ("dev",)),
                   Edge("d", ("id",), "r", ("dev",))],
            output_attrs=["device", "part", "supply"],
        )
        self.assert_matches_brute_force(spec)

    def test_multiplicity_and_duplicates(self):
        # 源表完全相同的多行，用 multiplicity 表示
        device = Table(
            [Column("id", "int64", False, "device")],
            [(1,), (2,)], multiplicities=[2, 3],
        )
        parts = Table(
            [Column("dev", "int64", False, "device"),
             Column("part", "text", False, "part")],
            [(1, "a"), (1, "b"), (2, "c")],
            multiplicities=[4, 1, 5],
        )
        supply = Table(
            [Column("dev", "int64", False, "device"),
             Column("sup", "text", False, "supply")],
            [(1, "x"), (2, "q"), (2, "q")],  # 重复行
            multiplicities=[1, 1, 1],
        )
        spec = Spec(
            tables={"device": device, "parts": parts, "supply": supply},
            instances={"d": "device", "l": "parts", "r": "supply"},
            edges=[Edge("d", ("id",), "l", ("dev",)),
                   Edge("d", ("id",), "r", ("dev",))],
            output_attrs=["device", "part", "supply"],
        )
        sess, store, _ = self.assert_matches_brute_force(spec)
        store.close()

    def test_null_join_never_matches(self):
        a = Table(
            [Column("k", "int64", True, "k"),
             Column("av", "text", True, "av")],
            [(1, "p"), (None, "n1"), (2, None)],
        )
        b = Table(
            [Column("k", "int64", True, "k"),
             Column("bv", "text", True, "bv")],
            [(1, "x"), (None, "n2"), (2, None)],
        )
        spec = Spec(
            tables={"a": a, "b": b},
            instances={"a": "a", "b": "b"},
            edges=[Edge("a", ("k",), "b", ("k",))],
            output_attrs=["k", "av", "bv"],
        )
        sess, store, bag = self.assert_matches_brute_force(spec)
        # null 键行被淘汰，只有 k=1 与 k=2 两组，共 2 行
        self.assertEqual(sess.count(), 2)
        # bv 为 null 的输出值仍可投影出来（null 只在 *连接键* 上不匹配）
        vals = [r for r, _ in bag]
        self.assertIn((2, None, None), vals)
        store.close()

    def test_empty_branch_zero_weight(self):
        a = Table([Column("k", "int64", False, "k"),
                   Column("av", "text", False, "av")],
                  [(1, "p"), (2, "q")])
        b = Table([Column("k", "int64", False, "k"),
                   Column("bv", "text", False, "bv")],
                  [(9, "x")])  # 无任何匹配 -> 空结果
        spec = Spec(
            tables={"a": a, "b": b},
            instances={"a": "a", "b": "b"},
            edges=[Edge("a", ("k",), "b", ("k",))],
            output_attrs=["k", "av", "bv"],
        )
        sess, store = build_session(spec)
        self.assertEqual(sess.count(), 0)
        self.assertEqual(sess.page(0, 10), [])
        self.assertEqual(brute_force_count(spec), 0)
        store.close()

    def test_composite_multi_column_key(self):
        # 多列分隔键 (k1, k2)，必须同时相等
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
        self.assert_matches_brute_force(spec)

    def test_role_reuse_and_self_source(self):
        # 同一来源表 emp 被两个角色 mgr / sub 复用（自身来源复用、身份不混）。
        # 复用同表时，非连接属性（team/name）也会在两角色出现，按运行交集
        # 它们必须连通，因此让连接经过 link，且 team/name 仅从 mgr 输出。
        emp = Table(
            [Column("eid", "int64", False, "person_id"),
             Column("team", "int64", False, "team"),
             Column("name", "text", False, "name")],
            [(1, 7, "boss"), (2, 7, "alice"), (3, 8, "bob")],
        )
        link = Table(
            [Column("m", "int64", False, "mgr_person"),
             Column("e", "int64", False, "sub_person")],
            [(1, 2), (1, 3)],
        )
        # 让 link 的物理列在逻辑属性上分别对齐两个角色的 person_id
        link2 = Table(
            [Column("m", "int64", False, "person_id_m"),
             Column("e", "int64", False, "person_id_s")],
            [(1, 2), (1, 3)],
        )
        mgr_tbl = Table(
            [Column("eid", "int64", False, "person_id_m"),
             Column("team", "int64", False, "team"),
             Column("name", "text", False, "mgr_name")],
            [(1, 7, "boss"), (2, 7, "alice"), (3, 8, "bob")],
        )
        sub_tbl = Table(
            [Column("eid", "int64", False, "person_id_s"),
             Column("team", "int64", False, "sub_team"),
             Column("name", "text", False, "sub_name")],
            [(1, 7, "boss"), (2, 7, "alice"), (3, 8, "bob")],
        )
        spec = Spec(
            tables={"mgr_tbl": mgr_tbl, "sub_tbl": sub_tbl, "link": link2},
            instances={"mgr": "mgr_tbl", "lnk": "link", "sub": "sub_tbl"},
            edges=[Edge("mgr", ("eid",), "lnk", ("m",)),
                   Edge("sub", ("eid",), "lnk", ("e",))],
            output_attrs=["team", "mgr_name", "person_id_s"],
        )
        sess, store, bag = self.assert_matches_brute_force(spec)
        # boss(person_id_m=1) 经 link 管理 person_id_s=2(alice) 与 3(bob)
        self.assertEqual(sess.count(), 2)
        self.assertEqual(sess.row_at(0), (7, "boss", 2))
        self.assertEqual(sess.row_at(1), (7, "boss", 3))
        store.close()

    def test_same_source_two_roles(self):
        # 同一来源表 node 被 parent / child 两个角色复用（自身来源复用、
        # 身份不混）。node 只含连接键 node_id；角色专属标签放在桥接表 edge
        # 上（parent_label/child_label），保证每个逻辑属性的出现节点连通。
        # 两条边各自做等值连接：parent 角色匹配 p、child 角色匹配 c，
        # 两角色独立取同一张 node 表的不同行，身份不会互相混淆。
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
        sess, store, bag = self.assert_matches_brute_force(spec)
        # 唯一桥接行：parent 角色匹配 id=1、child 角色独立匹配 id=2。
        # node_id 是经 parent-e-child 连通的共享属性，两角色身份独立、
        # 各从同源表取自己的匹配行，不会把 child 的 id=2 错当成 parent。
        self.assertEqual(sess.count(), 1)
        self.assertEqual(sess.row_at(0)[1:], ("root", "leaf"))
        store.close()

    def test_projection_merges_weights(self):
        # 投影掉连接键 device 后，不同内部节点产生相同输出 -> 重数相加、排序合并
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
        sess, store, bag = self.assert_matches_brute_force(spec)
        bag_dict = dict(bag)
        self.assertEqual(bag_dict[("a",)], 2)  # 两个设备都产生 part=a
        self.assertEqual(bag_dict[("b",)], 1)
        # 展开后的连续秩
        self.assertEqual(sess.page(0, 10), [("a",), ("a",), ("b",)])
        store.close()

    def test_ordering_null_int_text_bool(self):
        t = Table(
            [Column("i", "int64", True, "i"),
             Column("s", "text", True, "s"),
             Column("flag", "bool", True, "flag")],
            [
                (2, "b", True),
                (None, "a", False),
                (1, None, None),
                (1, "a", True),
                (None, "b", False),
            ],
        )
        spec = Spec(
            tables={"t": t},
            instances={"t": "t"},
            edges=[],
            output_attrs=["i", "s", "flag"],
        )
        self.assert_matches_brute_force(spec)

    def test_chain_three_levels(self):
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
        self.assert_matches_brute_force(spec)


class TestConditions(unittest.TestCase, CorrectnessMixin):
    def _spec(self):
        device = Table([Column("id", "int64", True, "device")],
                       [(1,), (2,), (None,)])
        parts = Table(
            [Column("dev", "int64", True, "device"),
             Column("part", "text", True, "part")],
            [(1, "a"), (1, "b"), (2, "a"), (None, "z")],
        )
        supply = Table(
            [Column("dev", "int64", False, "device"),
             Column("sup", "bool", False, "sup")],
            [(1, True), (1, False), (2, True)],
        )
        return Spec(
            tables={"device": device, "parts": parts, "supply": supply},
            instances={"d": "device", "l": "parts", "r": "supply"},
            edges=[Edge("d", ("id",), "l", ("dev",)),
                   Edge("d", ("id",), "r", ("dev",))],
            output_attrs=["device", "part", "sup"],
        )

    def test_conditions_renumber(self):
        spec = self._spec()
        sess, store = build_session(spec)
        try:
            cases = [
                [("device", "eq", 1)],
                [("device", "eq", 2)],
                [("part", "eq", "a")],
                [("sup", "eq", True)],
                [("device", "is_null", None)],
                [("part", "is_null", None)],
                [("device", "eq", 1), ("sup", "eq", False)],
                [("part", "eq", "a"), ("device", "eq", 2)],
            ]
            for raw in cases:
                conds = [
                    Condition.is_null(a) if op == "is_null"
                    else Condition.eq(a, v)
                    for a, op, v in raw
                ]
                view = sess.restricted(conds)
                expected = brute_force_filtered(spec, raw)
                exp_total = sum(m for _, m in expected)
                self.assertEqual(view.count(), exp_total, msg=f"条件 {raw}")
                exp_rows = []
                for tup, mult in expected:
                    exp_rows.extend([tup] * mult)
                got = [view.row_at(i) for i in range(exp_total)]
                self.assertEqual(got, exp_rows, msg=f"条件 {raw}")
                # 重新编号：筛选后第 0 秩必须是筛选结果首行，而非未筛选的第 0 行
                if exp_total:
                    self.assertEqual(view.page(0, exp_total), exp_rows)
                self.assertEqual(view.page(exp_total, 3), [])
                self.assertEqual(view.page(0, 0), [])
        finally:
            store.close()

    def test_bad_condition_rejected(self):
        spec = self._spec()
        sess, store = build_session(spec)
        try:
            with self.assertRaises(Exception):
                sess.restricted([Condition.eq("device", "not-an-int")])
            with self.assertRaises(Exception):
                sess.restricted([Condition.eq("nope", 1)])
            with self.assertRaises(Exception):
                sess.restricted([Condition.eq("device", None)])
        finally:
            store.close()


class TestValidation(unittest.TestCase):
    def _base(self):
        a = Table([Column("k", "int64", False, "k"),
                   Column("av", "text", False, "av")], [(1, "x")])
        b = Table([Column("k", "int64", False, "k"),
                   Column("bv", "text", False, "bv")], [(1, "y")])
        return a, b

    def test_cycle_or_disconnected_rejected(self):
        a, b = self._base()
        # 0 条边、两个实例 -> 不连通
        spec = Spec(
            tables={"a": a, "b": b},
            instances={"a": "a", "b": "b"},
            edges=[],
            output_attrs=["k", "av", "bv"],
        )
        with self.assertRaises(SpecError):
            spec.validate()

    def test_running_intersection_violation(self):
        # 同一逻辑属性 k 出现在 a 与 c，但中间节点 b 不含 k：
        # a-b 经 x1 连接、b-c 经 x2 连接，k 的出现节点 {a,c} 不连通。
        a = Table([Column("k", "int64", False, "k"),
                   Column("x1", "int64", False, "x1"),
                   Column("av", "text", False, "av")], [(1, 1, "x")])
        b = Table([Column("x1", "int64", False, "x1"),
                   Column("x2", "int64", False, "x2")], [(1, 2)])
        c = Table([Column("x2", "int64", False, "x2"),
                   Column("k", "int64", False, "k"),
                   Column("cv", "text", False, "cv")], [(2, 1, "z")])
        spec = Spec(
            tables={"a": a, "b": b, "c": c},
            instances={"a": "a", "b": "b", "c": "c"},
            edges=[
                Edge("a", ("x1",), "b", ("x1",)),
                Edge("b", ("x2",), "c", ("x2",)),
            ],
            output_attrs=["av", "cv"],
        )
        with self.assertRaises(SpecError):
            spec.validate()

    def test_type_mismatch_key_rejected(self):
        a = Table([Column("k", "int64", False, "k")], [(1,)])
        b = Table([Column("k", "text", False, "k")], [("1",)])
        spec = Spec(
            tables={"a": a, "b": b},
            instances={"a": "a", "b": "b"},
            edges=[Edge("a", ("k",), "b", ("k",))],
            output_attrs=["k"],
        )
        with self.assertRaises(SpecError):
            spec.validate()

    def test_incompatible_output_order_rejected(self):
        # 链 a-b-c，c 只匹配部分 b 行（存在空分支）。要求把最深层的 cv
        # 作为输出首列：提升 cv 时，含/不含 c 匹配的分支作用域不同构，
        # 无法在不展开组合的前提下重排为该全序，必须拒绝。
        a = Table([Column("ka", "int64", False, "ka"),
                   Column("av", "int64", False, "av")],
                  [(1, 10), (2, 20)])
        b = Table([Column("ka", "int64", False, "ka"),
                   Column("kb", "int64", False, "kb"),
                   Column("bv", "int64", False, "bv")],
                  [(1, 100, 30), (2, 200, 40)])
        c = Table([Column("kb", "int64", False, "kb"),
                   Column("cv", "int64", False, "cv")],
                  [(100, 999)])  # 仅 kb=100 有匹配，kb=200 是空分支
        spec = Spec(
            tables={"a": a, "b": b, "c": c},
            instances={"a": "a", "b": "b", "c": "c"},
            edges=[Edge("a", ("ka",), "b", ("ka",)),
                   Edge("b", ("kb",), "c", ("kb",))],
            output_attrs=["cv", "av", "bv"],  # 最深属性提到最前
        )
        store_ctx = tempfile.mkdtemp()
        path = os.path.join(store_ctx, "bad.db")
        store = DeliveryStore.create(path, spec)
        with self.assertRaises(BuildError):
            build_factorized(store, spec)
        store.close()


if __name__ == "__main__":
    unittest.main(verbosity=2)
