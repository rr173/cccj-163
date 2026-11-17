"""随机化属性测试：生成随机小树与数据，逐案对比因子交付与独立逐行枚举。

覆盖随机的：树形状（链/星）、列类型、可空、multiplicity、连接键、
投影列顺序（含会触发规范化的交错顺序与不可重排顺序）、条件组合。
"""
import os
import sys
import tempfile
import random
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from factor_delivery import (
    Column, Table, Edge, Spec,
    build_factorized, DeliveryStore, QuerySession, Condition,
    brute_force_join, brute_force_filtered,
)
from factor_delivery.builder import BuildError


def rand_value(rng, dtype, nullable):
    if nullable and rng.random() < 0.18:
        return None
    if dtype == "int64":
        return rng.randint(0, 6)
    if dtype == "text":
        return rng.choice(["a", "b", "c", "d"])
    return rng.choice([True, False])


class TestRandomized(unittest.TestCase):
    def test_random_chain_and_star(self):
        rng = random.Random(20261117)
        checked = 0
        builds_rejected = 0
        for trial in range(120):
            spec = self._make_spec(rng, trial)
            # 先看参考实现是否接受该规格
            try:
                spec.validate()
            except Exception:
                continue
            tmp = tempfile.mkdtemp()
            path = os.path.join(tmp, "r.db")
            try:
                store = DeliveryStore.create(path, spec)
                try:
                    build_factorized(store, spec)
                except BuildError:
                    builds_rejected += 1
                    store.close()
                    continue
                store.close()
                store = DeliveryStore.open(path)
                sess = QuerySession(store)
                self._check_one(spec, sess, rng)
                store.close()
                checked += 1
            finally:
                if os.path.exists(path):
                    try:
                        os.remove(path)
                    except OSError:
                        pass
        self.assertGreater(checked, 60)
        # 应当有一部分交错顺序被规范化拒绝，证明检测真实生效
        self.assertGreater(builds_rejected, 0)

    def _check_one(self, spec, sess, rng):
        bag = brute_force_join(spec)
        exp_total = sum(m for _, m in bag)
        self.assertEqual(sess.count(), exp_total)
        expanded = []
        for tup, mult in bag:
            expanded.extend([tup] * mult)
        # 随机抽秩核对
        for _ in range(min(20, max(1, exp_total))):
            r = rng.randrange(max(1, exp_total)) if exp_total else 0
            if r < exp_total:
                self.assertEqual(sess.row_at(r), expanded[r])
        # 小结果全量分页核对
        if exp_total <= 300:
            pages, r = [], 0
            while r < exp_total:
                chunk = sess.page(r, rng.randint(1, 7))
                self.assertTrue(chunk)
                pages.extend(chunk)
                r += len(chunk)
            self.assertEqual(pages, expanded)
        self.assertEqual(sess.page(exp_total, 3), [])
        self.assertEqual(sess.page(0, 0), [])
        # 随机条件
        for _ in range(3):
            conds_raw = self._random_conditions(spec, rng)
            conds = [
                Condition.is_null(a) if op == "is_null" else Condition.eq(a, v)
                for a, op, v in conds_raw
            ]
            view = sess.restricted(conds)
            expected = brute_force_filtered(spec, conds_raw)
            exp_c = sum(m for _, m in expected)
            self.assertEqual(view.count(), exp_c)
            if exp_c <= 200:
                exp_rows = []
                for tup, mult in expected:
                    exp_rows.extend([tup] * mult)
                got = [view.row_at(i) for i in range(exp_c)]
                self.assertEqual(got, exp_rows)
                self.assertEqual(view.page(exp_c, 3), [])

    def _random_conditions(self, spec, rng):
        out = []
        for attr in spec.output_attrs:
            if rng.random() < 0.35:
                nullable = spec.attr_nullable[attr]
                dtype = spec.attr_type[attr]
                if nullable and rng.random() < 0.4:
                    out.append((attr, "is_null", None))
                else:
                    v = rand_value(rng, dtype, False)
                    if v is not None:
                        out.append((attr, "eq", v))
        return out[:2]

    def _make_spec(self, rng, trial):
        shape = rng.choice(["chain", "star"])
        n = rng.randint(2, 4)
        # 每个实例的列：一个连接键 attr（共享 key_i）+ 0~2 个本地属性
        tables = {}
        instances = {}
        edges = []
        all_attrs = []
        for i in range(n):
            cols = []
            # 与父连接的键
            if i > 0:
                key_attr = f"key_{i}"
                cols.append(Column(f"k{i}", "int64", rng.random() < 0.3, key_attr))
            # 本地属性
            nlocal = rng.randint(0, 2)
            for j in range(nlocal):
                attr = f"t{i}_a{j}"
                dtype = rng.choice(["int64", "text", "bool"])
                cols.append(Column(attr, dtype, rng.random() < 0.3, attr))
                all_attrs.append(attr)
            rows, mults = [], []
            nrows = rng.randint(0, 6)
            for _ in range(nrows):
                rows.append(tuple(rand_value(rng, c.dtype, c.nullable) for c in cols))
                mults.append(rng.randint(1, 3))
            tname = f"tab{i}"
            tables[tname] = Table(cols, rows, mults)
            instances[f"u{i}"] = tname
        # 边：子节点的 key_i 与父节点新增的同名列
        # 父节点需要含该 key 属性 -> 在父表补一列
        for i in range(1, n):
            key_attr = f"key_{i}"
            parent = i - 1 if shape == "chain" else 0
            pname = tables and f"tab{parent}"
            pcols = tables[pname].columns
            if not any(c.attr == key_attr for c in pcols):
                # 给父表加键列并补值（重建表）
                new_col = Column(f"pk{i}", "int64",
                                 tables[f"tab{i}"].columns[0].nullable, key_attr)
                old = tables[pname]
                vals = []
                for row in old.rows:
                    vals.append(row + (rand_value(rng, "int64", new_col.nullable),))
                tables[pname] = Table(
                    old.columns + [new_col], vals, list(old.multiplicities)
                )
                all_attrs_insert = key_attr
            edges.append(Edge(f"u{parent}", (f"pk{i}",), f"u{i}", (f"k{i}",)))
        # 输出列：从所有出现过的属性里随机选并随机排序
        candidate = []
        for tname, tbl in tables.items():
            for c in tbl.columns:
                if c.attr not in candidate:
                    candidate.append(c.attr)
        rng.shuffle(candidate)
        output = candidate[:rng.randint(1, len(candidate))]
        return Spec(
            tables=tables,
            instances=instances,
            edges=edges,
            output_attrs=output,
            name=f"rand{trial}",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
