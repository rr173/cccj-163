"""主演练：设备 d 左 2 万部件 × 右 2 万供应点 = 4 亿逻辑行。

验证：
1. 因子结构规模随源表/键分组增长，文件中不存在 4 亿行宽表；
2. 精确总重数 = 4 亿；
3. 开头/中部/末尾小页面，并用整数除余独立推算左右位置；
4. d2 有不同候选，不能跨设备组合；
5. 投影掉设备后两设备产生相同输出值，重数相加且排序位置正确；
6. 记录真实因子节点数、文件大小、SQL 访问计数；
7. 重开交付文件后不依赖源数据仍可计数/取页/条件计数。
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from factor_delivery import (
    Column, Table, Edge, Spec,
    build_factorized, DeliveryStore, QuerySession, Condition,
)

NL, NR = 20000, 20000


def build_spec(project_device=True):
    # 设备表：d 与 d2
    device = Table(
        [Column("id", "text", False, "device")],
        [("d",), ("d2",)],
    )
    # 左部件：d 有 20000 个；d2 只有 3 个不同候选（证明不跨设备）
    left_rows = [("d", i) for i in range(NL)]
    left_rows += [("d2", 9001), ("d2", 9002), ("d2", 9003)]
    left = Table(
        [Column("dev", "text", False, "device"),
         Column("part", "int64", False, "part")],
        left_rows,
    )
    # 右供应点：d 有 20000 个；d2 只有 2 个
    right_rows = [("d", 100000 + j) for j in range(NR)]
    right_rows += [("d2", 777), ("d2", 888)]
    right = Table(
        [Column("dev", "text", False, "device"),
         Column("sup", "int64", False, "supply")],
        right_rows,
    )
    output = ["part", "supply"]
    if project_device:
        output = ["device"] + output
    return Spec(
        tables={"device": device, "left": left, "right": right},
        instances={"d": "device", "l": "left", "r": "right"},
        edges=[Edge("d", ("id",), "l", ("dev",)),
               Edge("d", ("id",), "r", ("dev",))],
        output_attrs=output,
        name="device_demo",
    )


def count_logical_rows_in_file(path):
    """审计：统计交付文件里所有物理表的行数，证明没有 4 亿行宽表。"""
    import sqlite3
    conn = sqlite3.connect(path)
    total = 0
    widest = 0
    for (name,) in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ):
        n = conn.execute(f"SELECT COUNT(*) FROM {name}").fetchone()[0]
        total += n
        widest = max(widest, n)
    conn.close()
    return total, widest


def main():
    tmp = tempfile.mkdtemp()
    path = os.path.join(tmp, "device_delivery.db")

    spec = build_spec(project_device=True)
    store = DeliveryStore.create(path, spec)
    stmts_before = store.conn.statements
    root_id = build_factorized(store, spec)
    build_stmts = store.conn.statements - stmts_before
    stats = store.read_build_stats()
    store.close()

    print("=" * 68)
    print("阶段一：构建因子结构（绝不展开 4 亿组合）")
    print("=" * 68)
    file_size = os.path.getsize(path)
    phys_total, phys_widest = count_logical_rows_in_file(path)
    print(f"交付文件: {path}")
    print(f"文件大小: {file_size:,} 字节 ({file_size / 1024 / 1024:.2f} MiB)")
    print(f"文件内所有物理表行数合计: {phys_total:,}（最大单表 {phys_widest:,}）")
    print(f"构建期 SQL 语句数: {build_stmts:,}")
    print("因子节点: "
          f"group={stats['node_group']} product={stats['node_product']} "
          f"leaf={stats['node_leaf']} empty={stats['node_empty']}")
    print(f"group entry 总数: {int(stats['group_entries']):,}")
    print(f"product child 总数: {stats['product_children']}")
    print(f"约减后各实例行数: "
          f"{ {k.split(':')[1]: v for k, v in stats.items() if k.startswith('reduced_rows:')} }")

    # 硬性审计：物理表行数、因子节点数都必须远小于 4 亿逻辑行，
    # 随源表/键分组规模增长，证明没有暗中物化宽连接结果。
    logical_rows = NL * NR
    node_total = sum(int(stats[k]) for k in
                     ("node_group", "node_product", "node_leaf",
                      "node_empty", "node_union")
                     if k in stats)
    assert phys_widest < logical_rows // 1000, "存在接近逻辑输出规模的物理表"
    assert phys_total < logical_rows // 1000, "物理总行数异常，疑似物化宽连接"
    assert node_total < logical_rows // 1000, "因子节点数异常，疑似展开组合"
    # 不存在任何一张表有 4 亿行；最大表也只在源表行数 + 因子 entry 的量级
    assert phys_widest <= (NL + NR) * 2 + 100
    print(f"审计通过：逻辑输出 {logical_rows:,} 行，"
          f"因子节点合计仅 {node_total}，最大物理表 {phys_widest:,} 行")

    # 重开：仅依赖交付文件
    store = DeliveryStore.open(path)
    sess = QuerySession(store)

    print()
    print("=" * 68)
    print("阶段二：重开文件，精确总重数与小页面取秩")
    print("=" * 68)
    total = sess.count()
    expect_total = NL * NR + 3 * 2
    print(f"精确总重数: {total:,}  期望: {expect_total:,}  "
          f"匹配={total == expect_total}")
    assert total == expect_total

    # d 设备的 4 亿行在排序中位于 d2 之前（text: 'd' < 'd2'）
    d_block = NL * NR
    for label, r in (("开头", 0), ("中部", d_block // 2),
                     ("d 块末尾", d_block - 2)):
        page = sess.page(r, 2)
        # 独立除余推算：d 块内秩 k，part 随 k // NR，supply 随 k % NR
        k = r
        exp_part = k // NR
        exp_sup = 100000 + (k % NR)
        print(f"\n[{label}] 秩 r={r:,} 的页: {page}")
        print(f"  除余校验: part={exp_part} (={r}//{NR}), "
              f"supply={exp_sup} (100000+{r}%{NR})")
        assert page[0] == ("d", exp_part, exp_sup)

    # 全局末尾（d2 块）
    last_r = total - 1
    last_page = sess.page(last_r, 1)
    print(f"\n[全局末尾] 秩 r={last_r:,}: {last_page}")
    assert last_page[0][0] == "d2"

    print()
    print("=" * 68)
    print("阶段三：d2 隔离——不能跨设备组合")
    print("=" * 68)
    view_d2 = sess.restricted([Condition.eq("device", "d2")])
    print(f"device=d2 重数: {view_d2.count()}（期望 3*2=6）")
    assert view_d2.count() == 6
    d2_rows = view_d2.page(0, 10)
    print("d2 全部行:")
    for row in d2_rows:
        print("  ", row)
    parts = {row[1] for row in d2_rows}
    sups = {row[2] for row in d2_rows}
    assert parts == {9001, 9002, 9003}
    assert sups == {777, 888}
    # d2 的 part 绝不会出现在 d 的取值范围 0..19999 之外的串扰
    view_d = sess.restricted([Condition.eq("device", "d")])
    assert view_d.count() == d_block

    print()
    print("=" * 68)
    print("阶段四：条件后重新编号（不是从未筛选 r 截取）")
    print("=" * 68)
    v = sess.restricted([Condition.eq("part", 5)])
    print(f"part=5 重数: {v.count()}（d 下每个 part 配 20000 供应点）")
    assert v.count() == NR
    first = v.row_at(0)
    print(f"筛选后第 0 秩: {first}（应为 part=5 的最小 supply，连续重编号）")
    assert first == ("d", 5, 100000)
    assert v.page(v.count(), 1) == []

    query_stmts = store.conn.statements
    print(f"\n查询阶段累计 SQL 语句数: {query_stmts:,}（与 4 亿行数无关）")
    store.close()

    print()
    print("=" * 68)
    print("阶段五：投影掉设备——相同输出值重数相加、排序合并")
    print("=" * 68)
    proj_path = os.path.join(tmp, "device_proj.db")
    pspec = build_spec(project_device=False)
    pstore = DeliveryStore.create(proj_path, pspec)
    build_factorized(pstore, pspec)
    pstats = pstore.read_build_stats()
    pstore.close()
    pstore = DeliveryStore.open(proj_path)
    psess = QuerySession(pstore)
    print(f"大规模投影后总重数（与带设备列一致）: {psess.count():,}")
    assert psess.count() == expect_total
    head = psess.page(0, 3)
    print("大规模投影后开头三行:", head)
    tail = psess.page(psess.count() - 3, 3)
    print("大规模投影后末尾三行:", tail)
    print(f"投影因子节点: group={pstats['node_group']} "
          f"product={pstats['node_product']} union={pstats.get('node_union')} "
          f"leaf={pstats['node_leaf']}")
    pstore.close()

    # 精确的小规模重叠实验：d 与 d2 在投影掉设备后产生 *完全相同* 的输出值
    overlap_path = os.path.join(tmp, "overlap_proj.db")
    ospec = build_overlap_spec()
    ostore = DeliveryStore.create(overlap_path, ospec)
    build_factorized(ostore, ospec)
    ostat = ostore.read_build_stats()
    ostore.close()
    ostore = DeliveryStore.open(overlap_path)
    osess = QuerySession(ostore)
    print()
    print("-- 重叠实验：d 与 d2 各自产生 (part=1, supply=9) 与 (part=2, supply=9) --")
    # 带设备列时每个组合各 1 份，共 4 个不同的带键元组
    print("投影后完整袋（按全序，重复元组各占一秩）:")
    rows = osess.page(0, osess.count())
    for r in rows:
        print("   ", r)
    # 投影掉设备后 (1,9) 与 (2,9) 各由两个设备贡献 -> 重数各为 2
    from factor_delivery.reference import brute_force_join
    bag = dict(brute_force_join(ospec))
    print("逐行枚举核对的合并重数:", bag)
    assert bag[(1, 9)] == 2 and bag[(2, 9)] == 2
    assert osess.count() == 4
    assert rows == [(1, 9), (1, 9), (2, 9), (2, 9)]
    # 排序位置：秩 0、1 是合并后的 (1,9)，秩 2、3 是 (2,9)
    assert osess.row_at(0) == (1, 9) and osess.row_at(1) == (1, 9)
    assert osess.row_at(2) == (2, 9) and osess.row_at(3) == (2, 9)
    print(f"重叠实验因子节点: group={ostat['node_group']} "
          f"product={ostat['node_product']} union={ostat.get('node_union')} "
          f"leaf={ostat['node_leaf']}（union 保留两设备分支，未做跨设备笛卡尔积）")
    ostore.close()

    print()
    print("全部演练断言通过。")


def build_overlap_spec():
    """两设备投影后产生相同 (part, supply) 的小规格，用于核对重数相加。"""
    device = Table(
        [Column("id", "text", False, "device")],
        [("d",), ("d2",)],
    )
    left = Table(
        [Column("dev", "text", False, "device"),
         Column("part", "int64", False, "part")],
        [("d", 1), ("d", 2), ("d2", 1), ("d2", 2)],
    )
    right = Table(
        [Column("dev", "text", False, "device"),
         Column("sup", "int64", False, "supply")],
        [("d", 9), ("d2", 9)],
    )
    return Spec(
        tables={"device": device, "left": left, "right": right},
        instances={"d": "device", "l": "left", "r": "right"},
        edges=[Edge("d", ("id",), "l", ("dev",)),
               Edge("d", ("id",), "r", ("dev",))],
        output_attrs=["part", "supply"],  # 刻意不含 device
        name="overlap_projection",
    )


if __name__ == "__main__":
    main()
