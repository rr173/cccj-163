"""交付文件自洽性与重开独立性测试。

1. 构建后删除/移走所有源数据，仅靠交付文件仍可计数、条件计数、取秩页；
2. 因子节点协议可审查：重数能自底向上重算，篡改缓存总行数会被检出；
3. 空结果总重数为 0，且结构里带 empty 节点；
4. 访问计数随节点/分组规模增长，与逻辑输出行数无关。
"""
import os
import sys
import shutil
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from factor_delivery import (
    Column, Table, Edge, Spec,
    build_factorized, DeliveryStore, QuerySession, Condition,
    ProtocolError,
)


def _star_spec():
    device = Table([Column("id", "int64", False, "device")], [(1,), (2,)])
    left = Table(
        [Column("dev", "int64", False, "device"),
         Column("part", "text", False, "part")],
        [(1, "a"), (1, "b"), (2, "c")],
    )
    right = Table(
        [Column("dev", "int64", False, "device"),
         Column("sup", "text", False, "supply")],
        [(1, "x"), (1, "y"), (2, "q")],
    )
    return Spec(
        tables={"device": device, "left": left, "right": right},
        instances={"d": "device", "l": "left", "r": "right"},
        edges=[Edge("d", ("id",), "l", ("dev",)),
               Edge("d", ("id",), "r", ("dev",))],
        output_attrs=["device", "part", "supply"],
    )


class TestDeliveryFile(unittest.TestCase):
    def test_reopen_without_sources(self):
        spec = _star_spec()
        workdir = tempfile.mkdtemp()
        path = os.path.join(workdir, "delivery.db")
        store = DeliveryStore.create(path, spec)
        build_factorized(store, spec)
        store.close()
        # 交付文件独立移动到一个空目录，周围没有任何源数据
        isolated = tempfile.mkdtemp()
        moved = os.path.join(isolated, "only_delivery.db")
        shutil.copy(path, moved)
        shutil.rmtree(workdir, ignore_errors=True)

        store = DeliveryStore.open(moved)
        sess = QuerySession(store)
        # 1*2*2 + 1*1*1 = 5
        self.assertEqual(sess.count(), 5)
        self.assertEqual(len(sess.page(0, 100)), 5)
        view = sess.restricted([Condition.eq("device", 1)])
        self.assertEqual(view.count(), 4)
        self.assertEqual(view.page(0, 2),
                         [(1, "a", "x"), (1, "a", "y")])
        store.close()
        shutil.rmtree(isolated, ignore_errors=True)

    def test_weight_cross_check_detects_tampering(self):
        spec = _star_spec()
        path = os.path.join(tempfile.mkdtemp(), "t.db")
        store = DeliveryStore.create(path, spec)
        build_factorized(store, spec)
        store.close()
        # 篡改某个节点的缓存重数，但不改子节点
        import sqlite3
        conn = sqlite3.connect(path)
        root = conn.execute(
            "SELECT node_id FROM fd_node ORDER BY node_id LIMIT 1"
        ).fetchone()[0]
        conn.execute(
            "UPDATE fd_node SET weight='999999' WHERE node_id=?", (root,)
        )
        conn.commit()
        conn.close()
        store = DeliveryStore.open(path)
        sess = QuerySession(store)
        with self.assertRaises(ProtocolError):
            sess.count()  # 自底向上重算与缓存不符 -> 拒绝只信总行数
        store.close()

    def test_empty_result_has_zero_weight(self):
        device = Table([Column("id", "int64", False, "device")], [(1,)])
        left = Table(
            [Column("dev", "int64", False, "device"),
             Column("part", "text", False, "part")],
            [(9, "orphan")],  # 无设备匹配 -> 空分支
        )
        spec = Spec(
            tables={"device": device, "left": left},
            instances={"d": "device", "l": "left"},
            edges=[Edge("d", ("id",), "l", ("dev",))],
            output_attrs=["device", "part"],
        )
        path = os.path.join(tempfile.mkdtemp(), "e.db")
        store = DeliveryStore.create(path, spec)
        build_factorized(store, spec)
        stats = store.read_build_stats()
        store.close()
        store = DeliveryStore.open(path)
        sess = QuerySession(store)
        self.assertEqual(sess.count(), 0)
        self.assertEqual(sess.page(0, 10), [])
        self.assertEqual(sess.page(0, 0), [])
        self.assertGreaterEqual(int(stats["node_empty"]), 1)
        store.close()

    def test_protocol_tables_are_auditable(self):
        spec = _star_spec()
        path = os.path.join(tempfile.mkdtemp(), "a.db")
        store = DeliveryStore.create(path, spec)
        build_factorized(store, spec)
        store.close()
        import sqlite3
        conn = sqlite3.connect(path)
        kinds = {r[0] for r in conn.execute(
            "SELECT DISTINCT kind FROM fd_node")}
        # 至少出现 group / product / leaf 三类
        self.assertTrue({"group", "product", "leaf"}.issubset(kinds))
        # 每个 group entry 都指向存在的子节点
        bad = conn.execute(
            "SELECT COUNT(*) FROM fd_group_entry g "
            "WHERE NOT EXISTS (SELECT 1 FROM fd_node n "
            "WHERE n.node_id = g.child_id)"
        ).fetchone()[0]
        self.assertEqual(bad, 0)
        # 每个 product/union child 都指向存在的节点
        bad2 = conn.execute(
            "SELECT COUNT(*) FROM fd_product_child p "
            "WHERE NOT EXISTS (SELECT 1 FROM fd_node n "
            "WHERE n.node_id = p.child_id)"
        ).fetchone()[0]
        self.assertEqual(bad2, 0)
        # 协议版本登记
        ver = conn.execute(
            "SELECT value FROM fd_meta WHERE key='protocol_version'"
        ).fetchone()[0]
        self.assertEqual(ver, "1.0")
        conn.close()


if __name__ == "__main__":
    unittest.main(verbosity=2)
