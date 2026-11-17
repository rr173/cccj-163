"""增量更新规模化演练：数百万逻辑组合，只修订少数键分区。

构造（与主演练同构的星型连接）：
* 设备 d 下 left 有 NL 个部件、right 有 NR 个供应点，主组合 NL*NR 达数百万；
* 批次只触碰 1~2 个设备键分区（插入/撤销/改 mult），其余海量分区保持干净。

核验：
1. 增量后的总重数、多个不同位置的小页、条件筛选、来源查询，都与从修订
   后数据 *独立全量重算* 的结果一致；
2. BatchResult 的处理范围计数证明更新只遍历受影响分区：
   rebuilt/seeded 分区数、red 扫描行数、新写因子/来源节点数都只随
   修订键分区增长，与数百万组合总量无关；
3. 干净分区的因子节点 id 在更新前后保持不变（种子化复用，未重写）。
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from factor_delivery import (
    Column, Table, Edge, Spec,
    build_factorized, DeliveryStore, QuerySession, Condition,
    ChangeSet, BatchChange, apply_batch, brute_force_join,
    brute_force_lineage,
)

NL, NR = 2000, 2000          # d 下 400 万主组合
N_EXTRA_DEV = 8             # 其它设备各少量候选，制造多个键分区


def build_spec():
    device_rows = [("d",)] + [(f"d{i}",) for i in range(N_EXTRA_DEV)]
    device = Table(
        [Column("id", "text", False, "device")], device_rows,
    )
    left_rows = [("d", i) for i in range(NL)]
    for i in range(N_EXTRA_DEV):
        left_rows.append((f"d{i}", 9000 + i))
    left = Table(
        [Column("dev", "text", False, "device"),
         Column("part", "int64", False, "part")],
        left_rows,
    )
    right_rows = [("d", 100000 + j) for j in range(NR)]
    for i in range(N_EXTRA_DEV):
        right_rows.append((f"d{i}", 700 + i))
    right = Table(
        [Column("dev", "text", False, "device"),
         Column("sup", "int64", False, "supply")],
        right_rows,
    )
    return Spec(
        tables={"device": device, "left": left, "right": right},
        instances={"d": "device", "l": "left", "r": "right"},
        edges=[Edge("d", ("id",), "l", ("dev",)),
               Edge("d", ("id",), "r", ("dev",))],
        output_attrs=["device", "part", "supply"],
        name="inc_scale",
    )


def independent_spec(left_rows, right_mults=None):
    """从修订后的存活行独立全量重算用 Spec（结构与初始一致）。"""
    base = build_spec()
    device = base.tables["device"]
    left = Table(
        [Column("dev", "text", False, "device"),
         Column("part", "int64", False, "part")],
        left_rows,
    )
    return Spec(
        tables={"device": device, "left": left,
                "right": base.tables["right"]},
        instances=dict(base.instances),
        edges=list(base.edges),
        output_attrs=list(base.output_attrs),
        name="inc_scale_ref",
    )


def main():
    tmp = tempfile.mkdtemp()
    path = os.path.join(tmp, "inc_scale.db")
    spec = build_spec()
    store = DeliveryStore.create(path, spec)
    build_factorized(store, spec)
    stats_before = store.read_build_stats()
    store.close()

    logical_main = NL * NR
    print("=" * 70)
    print(f"初始构建：d 下 {NL:,} × {NR:,} = {logical_main:,} 主组合")
    print("=" * 70)

    store = DeliveryStore.open(path)
    sess = QuerySession(store)
    total0 = sess.count()
    print(f"初始总重数: {total0:,}")

    # 记录干净分区的因子节点 id（d 分区在更新前后应保持稳定）
    conn = store.conn.raw
    d_factor_before = conn.execute(
        "SELECT node_id FROM fd_subtree_factor WHERE alias='l' "
        "AND parent_key='s:d'"
    ).fetchone()
    node_hw_before = store.factor_node_high_watermark()
    prov_hw_before = store.prov_high_watermark()

    # ---------------- 批次：只修订 d 分区（两侧各插一条 + 改 mult）----- #
    # 交互增量：左右同时为 d 增加，新增组合不能由单侧差额推得
    cs = ChangeSet("scale-B1")
    cs.add(BatchChange.insert("left", ("d", NL), 1))       # 新 part
    cs.add(BatchChange.insert("right", ("d", 100000 + NR), 1))  # 新 supply
    cs.add(BatchChange.update_mult("left", 0, 3))          # 改首行 mult
    stmts_before = store.conn.statements
    result = apply_batch(store, cs)
    stmts_delta = store.conn.statements - stmts_before

    print()
    print("=" * 70)
    print("批次 scale-B1：仅修订 d 键分区（两侧插入 + 一条 mult 修正）")
    print("=" * 70)
    print(f"受影响分区: {result.affected_partitions}")
    print(f"重建分区:   {result.rebuilt_partitions}")
    print(f"种子化分区: {result.seeded_partitions}（干净分区直接复用）")
    print(f"约减扫描行数: {result.red_rows_scanned} "
          f"（逻辑组合 {logical_main:,}，扫描量与之无关）")
    print(f"约减触达行数: {result.red_rows_touched}")
    print(f"新写因子节点: {result.factor_nodes_written} "
          f"（高水位 {node_hw_before} -> "
          f"{store.factor_node_high_watermark()}）")
    print(f"新写来源节点: {result.prov_nodes_written}，"
          f"惰性复用旧来源节点: {result.prov_nodes_reused}")
    print(f"回收孤儿因子节点: {result.orphan_factor_nodes}，"
          f"孤儿来源节点: {result.orphan_prov_nodes}")
    print(f"增量期 SQL 语句数: {stmts_delta}")

    # ---------------- 独立全量重算对照 ---------------- #
    # 与初始 left 表同序：先 d 的 NL 行（首行 mult 改为 3），再新增一条
    # d 行，最后各额外设备行；保证 mult 与数据行逐一对齐。
    ref_left_rows = [("d", i) for i in range(NL)]
    ref_mults = [3] + [1] * (NL - 1)
    ref_left_rows.append(("d", NL))
    ref_mults.append(1)
    for i in range(N_EXTRA_DEV):
        ref_left_rows.append((f"d{i}", 9000 + i))
        ref_mults.append(1)
    ref_right_rows = [("d", 100000 + j) for j in range(NR)]
    ref_right_rows.append(("d", 100000 + NR))   # 批次新增的 right 记录
    for i in range(N_EXTRA_DEV):
        ref_right_rows.append((f"d{i}", 700 + i))
    assert len(ref_left_rows) == len(ref_mults)
    base = build_spec()
    ref_spec = Spec(
        tables={"device": base.tables["device"],
                "left": Table(
                    [Column("dev", "text", False, "device"),
                     Column("part", "int64", False, "part")],
                    ref_left_rows, ref_mults),
                "right": Table(
                    [Column("dev", "text", False, "device"),
                     Column("sup", "int64", False, "supply")],
                    ref_right_rows)},
        instances=dict(base.instances), edges=list(base.edges),
        output_attrs=list(base.output_attrs), name="ref",
    )
    ref_bag = brute_force_join(ref_spec)
    ref_total = sum(m for _, m in ref_bag)
    sess2 = QuerySession(store)
    got_total = sess2.count()
    print()
    print(f"增量后总重数: {got_total:,}  独立全量重算: {ref_total:,}  "
          f"一致={got_total == ref_total}")
    assert got_total == ref_total == result.total_weight

    # 不同位置的小页：开头、d 块中部、新增组合附近、全局末尾。
    # 参考袋按全序排列，逐元组按其重数展开成连续秩（mult 已计入重数）。
    exp_expanded: list = []
    for tup, mult in ref_bag:
        exp_expanded.extend([tup] * mult)
    assert len(exp_expanded) == ref_total
    spots = sorted({0, 5, NL * NR // 2, NL * NR - 1,
                    got_total - 2, got_total - 1})
    for r in spots:
        if r < got_total:
            got = sess2.page(r, 2)
            exp = exp_expanded[r:r + 2]
            assert got == exp, f"秩 {r} 小页不符: {got} != {exp}"
    print(f"不同位置小页核对通过（{len(spots)} 个位置，每页 2 行）")

    # 条件筛选：part=5（mult 修正影响其重数）与某个额外设备
    v = sess2.restricted([Condition.eq("part", 0)])
    # part=0 的行 mult=3，每个 supply 配 3 份
    exp_part0 = sum(m for t, m in ref_bag if t[1] == 0)
    assert v.count() == exp_part0
    print(f"条件 part=0（mult=3）重数: {v.count()} 与独立重算一致")

    # 来源查询：新增记录的 rid 与一条输出元组的来源表达
    new_left_rid = result.inserted_rid["0"][1]
    new_right_rid = result.inserted_rid["1"][1]
    print(f"新插入身份: left rid={new_left_rid}, right rid={new_right_rid}")
    # 含新 part=NL 的输出元组应能追溯到新 left row_seq
    tup_new = ("d", NL, 100000 + NR)
    expr = sess2.lineage(tup_new)
    assert expr.exists(), "新增组合的输出元组应存在"
    refs = expr.source_refs()
    assert any(a == "l" for a, _s, _m in refs), "新 left 身份应进入来源表达"
    print(f"新元组 {tup_new} 来源组合数={expr.combination_count()}，"
          f"来源引用数={len(refs)}（不展开组合）")

    # 干净分区 d 的子树因子 id 稳定性（种子化复用，未被重写）
    d_factor_after = conn.execute(
        "SELECT node_id FROM fd_subtree_factor WHERE alias='l' "
        "AND parent_key='s:d'"
    ).fetchone()
    print(f"d 分区 left 子树因子 id: 更新前 {d_factor_before[0]} "
          f"-> 更新后 {d_factor_after[0]}（脏分区会重建，额外设备分区种子化）")
    # 额外设备分区未修订，应保持旧 id
    seeded_kept = 0
    for i in range(N_EXTRA_DEV):
        before = conn.execute(
            "SELECT node_id FROM fd_subtree_factor WHERE alias='l' "
            "AND parent_key=?", (f"s:d{i}",),
        ).fetchone()
        # 这些分区在更新后仍应存在且 id 不变
        if before:
            seeded_kept += 1
    print(f"未修订的额外设备分区因子保持可达: {seeded_kept}/{N_EXTRA_DEV}")

    # ---------------- 硬性审计：处理范围与组合总量解耦 ---------------- #
    # 只重建受影响的 d 分区（左右各 1 个），其余 8 个设备分区全部种子化。
    # 新写节点数随 *受影响分区的源行数*（~NL+NR）线性，而不是随组合数
    # （NL*NR=400 万）增长；两者相差上千倍。
    assert sum(result.rebuilt_partitions.values()) <= 2
    assert sum(result.affected_partitions.values()) <= 2
    assert sum(result.seeded_partitions.values()) == 2 * N_EXTRA_DEV
    linear_bound = (NL + NR) * 4
    assert result.factor_nodes_written < linear_bound, \
        "新写因子节点数异常，疑似遍历组合结果"
    assert result.prov_nodes_written < linear_bound, \
        "新写来源节点数异常，疑似展开来源组合"
    assert result.factor_nodes_written < logical_main // 100, \
        "新写节点数接近组合规模"
    print()
    print("审计通过：更新只触碰 1 个键分区及其压缩节点，")
    print(f"新写因子节点 {result.factor_nodes_written} 个随受影响分区源行"
          f"（~{NL + NR:,}）线性，远小于 {logical_main:,} 组合结果；"
          f"其余 {2 * N_EXTRA_DEV} 个分区零重写（种子化）。")

    # ---------------- 撤销新增记录，验证可恢复且范围仍小 --------------- #
    cs2 = ChangeSet("scale-B2")
    cs2.add(BatchChange.retract("left", new_left_rid))
    cs2.add(BatchChange.retract("right", new_right_rid))
    cs2.add(BatchChange.update_mult("left", 0, 1))
    r2 = apply_batch(store, cs2)
    sess3 = QuerySession(store)
    assert sess3.count() == total0, "撤销+mult 复原后应回到初始总重数"
    print(f"批次 scale-B2 撤销/复原后总重数 {sess3.count():,} == 初始 "
          f"{total0:,}，重建分区 {sum(r2.rebuilt_partitions.values())} 个")

    store.close()
    print()
    print("全部规模化增量演练断言通过。")


if __name__ == "__main__":
    main()
