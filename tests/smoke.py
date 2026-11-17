"""冒烟测试：最小三表连接走通构建、重开、计数、取页、条件。"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from factor_delivery import (
    Column, Table, Edge, Spec,
    build_factorized, DeliveryStore, QuerySession, Condition,
    brute_force_join, brute_force_count,
)


def main():
    # d 设备表：d, (键 k_d)
    device = Table(
        columns=[Column("id", "int64", False, "device")],
        rows=[(1,), (2,)],
    )
    # 左部件表：device 键 + part
    parts = Table(
        columns=[
            Column("dev", "int64", False, "device"),
            Column("part", "text", False, "part"),
        ],
        rows=[(1, "a"), (1, "b"), (2, "c"), (3, "z")],  # 3 是无效分支
    )
    # 右供应点表：device 键 + supply，含 null 键行（不匹配）
    supply = Table(
        columns=[
            Column("dev", "int64", True, "device"),
            Column("sup", "text", False, "supply"),
        ],
        rows=[(1, "x"), (1, "y"), (2, "q"), (None, "n")],
    )
    spec = Spec(
        tables={"device": device, "parts": parts, "supply": supply},
        instances={"d": "device", "l": "parts", "r": "supply"},
        edges=[
            Edge("d", ("id",), "l", ("dev",)),
            Edge("d", ("id",), "r", ("dev",)),
        ],
        output_attrs=["device", "part", "supply"],
        name="smoke",
    )

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "smoke.db")
        store = DeliveryStore.create(path, spec)
        root_id = build_factorized(store, spec)
        stats = store.read_build_stats()
        store.close()

        store = DeliveryStore.open(path)
        sess = QuerySession(store)
        total = sess.count()
        print("root:", root_id, "total:", total)
        print("all rows:")
        for row in sess.page(0, total):
            print(" ", row)
        print("page r=2 n=2:", sess.page(2, 2))
        print("out of range:", sess.page(999, 5))
        print("n=0:", sess.page(0, 0))
        view = sess.restricted([Condition.eq("device", 1)])
        print("device=1 count:", view.count())
        print("device=1 pages:", [view.page(i, 1) for i in range(view.count())])
        view2 = sess.restricted([Condition.is_null("device")])
        print("device is_null count:", view2.count())
        store.close()

    print("brute force count:", brute_force_count(spec))
    for tup, mult in brute_force_join(spec):
        print("  bf:", tup, mult)


if __name__ == "__main__":
    main()
