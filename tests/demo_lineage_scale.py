"""来源追溯规模化演练：单条输出元组承载上亿种来源组合。

构造（投影掉连接键 device，只输出 part）：
* 设备 d 有 NL 个部件行、NR 个供应点行；
* 投影后 d 的所有部件行产出同一个输出值（part 取常量），其来源表达是
  alt(NL 个部件来源) × alt(NR 个供应点来源) × 设备来源，
  来源组合数 = NL * NR（再乘 multiplicity），轻松达到上亿；
* 表达节点数只随来源行数（NL + NR + 少量）增长，与组合数无关；
* participates() 沿 DAG 判断，不展开组合，SQL 访问数有界；
* 关闭后在 *新进程* 中重开文件，内存 Spec / 输入文件均不可用，仍可追溯。

另含一个 multiplicity 放大场景，验证 mult 不逐份建节点、组合数仍精确。
"""
import os
import sys
import subprocess
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from factor_delivery import (
    Column, Table, Edge, Spec,
    build_factorized, DeliveryStore, QuerySession,
    brute_force_lineage,
)

NL = 20000
NR = 20000
NL2 = 5000
NR2 = 5000
# 目标：单条输出元组承载 d 组 2 万 × 2 万 = 4 亿 + d2 组 5 千 × 5 千 = 2.5 千万
# 种来源组合（d 组再乘设备 multiplicity=3），合计远超亿级。
TARGET_COMBOS_D = NL * NR          # 4 亿
TARGET_COMBOS_D2 = NL2 * NR2       # 2.5 千万


def build_scale_spec():
    # 两个设备 d / d2，各自有 NL/NL2 个部件行、NR/NR2 个供应点行；
    # 所有部件行的 part 取同一常量，投影 device 后两个 *连接键分组* 折叠到
    # 同一输出元组 -> 来源表达顶部是 alt(两个 prod 分支)，分组间不交叉组合。
    device = Table(
        [Column("id", "text", False, "device")],
        [("d",), ("d2",)],
        multiplicities=[3, 1],
    )
    left = Table(
        [Column("dev", "text", False, "device"),
         Column("part", "int64", False, "part")],
        [("d", 7) for _ in range(NL)]
        + [("d2", 7) for _ in range(NL2)],
    )
    right = Table(
        [Column("dev", "text", False, "device"),
         Column("sup", "int64", False, "supply")],
        [("d", 9) for _ in range(NR)]
        + [("d2", 9) for _ in range(NR2)],
    )
    return Spec(
        tables={"device": device, "left": left, "right": right},
        instances={"d": "device", "l": "left", "r": "right"},
        edges=[Edge("d", ("id",), "l", ("dev",)),
               Edge("d", ("id",), "r", ("dev",))],
        output_attrs=["part"],  # 刻意投影掉 device 与 supply
        name="lineage_scale",
    )


def main():
    tmp = tempfile.mkdtemp()
    path = os.path.join(tmp, "lineage_scale.db")

    spec = build_scale_spec()
    store = DeliveryStore.create(path, spec)
    stmts_before = store.conn.statements
    build_factorized(store, spec)
    build_stmts = store.conn.statements - stmts_before
    stats = store.read_build_stats()
    store.close()

    print("=" * 72)
    print("阶段一：构建（不展开来源组合）")
    print("=" * 72)
    size = os.path.getsize(path)
    print(f"交付文件: {path}")
    print(f"文件大小: {size:,} 字节 ({size / 1024 / 1024:.2f} MiB)")
    print(f"构建期 SQL 语句数: {build_stmts:,}")
    print("来源节点: src={src} prod={prod} alt={alt}".format(
        src=stats["prov_src"], prod=stats["prov_prod"], alt=stats["prov_alt"]))
    print(f"来源 src 行登记数: {stats['prov_src_rows']}（= 来源行规模，非组合数）")
    print(f"来源复合分支数: {stats['prov_comp_rows']}")

    # 重开：仅依赖交付文件
    store = DeliveryStore.open(path)
    sess = QuerySession(store)

    print()
    print("=" * 72)
    print("阶段二：单条元组承载 4 亿来源组合，表达节点有界")
    print("=" * 72)
    only_tuple = (7,)
    expr = sess.lineage(only_tuple)
    assert expr.exists()
    # d 组组合乘设备 mult=3；d2 组设备 mult=1；两键分组 alt 相加、不交叉
    expected_combos = TARGET_COMBOS_D * 3 + TARGET_COMBOS_D2
    print(f"输出元组 {only_tuple} 的来源组合数: "
          f"{expr.combination_count():,}（期望 {expected_combos:,}）")
    assert expr.combination_count() == expected_combos
    # 与结果袋重数一致
    assert sess.count() == expected_combos
    n_nodes = expr.node_count()
    n_persist = expr.persisted_node_count()
    print(f"表达节点数: {n_nodes}（持久化 {n_persist} / "
          f"会话内 {expr.virtual_node_count()}）")
    # 顶部必须是 alt（两个连接键分组折叠到同一元组），分组来源不交叉
    assert expr.root.kind == "alt", expr.root.kind
    print(f"顶部节点为 alt，{len(expr.root.children)} 个互斥连接分支，"
          f"无跨键笛卡尔积")
    # 硬性有界性：节点数必须是来源行量级，远小于组合数
    source_rows = NL + NR + NL2 + NR2 + 2
    assert n_nodes <= source_rows + 50, (
        f"表达节点 {n_nodes} 超出来源行量级 {source_rows}"
    )
    assert n_nodes < TARGET_COMBOS_D // 1000, "表达节点随组合数膨胀，疑似展开"
    print(f"有界性通过：{expected_combos:,} 组合仅用 {n_nodes} 个表达节点 "
          f"（来源行 {source_rows}）")

    # src 引用清单只是不同来源行，数量有界且不随 mult 展开
    refs = expr.source_refs()
    print(f"不同来源行引用数: {len(refs)}（d 设备 mult=3 仍是 1 个 src 行节点）")
    assert len(refs) == NL + NR + NL2 + NR2 + 2
    d_ref = [r for r in refs if r[0] == "d"]
    assert ("d", 0, 3) in d_ref and ("d", 1, 1) in d_ref

    print()
    print("=" * 72)
    print("阶段三：参与查询不展开组合（SQL 访问数有界）")
    print("=" * 72)
    # 选一个 d 组部件行与一个 d2 组部件行
    stmts0 = store.conn.statements
    assert expr.participates("l", 0) is True
    assert expr.participates("l", NL // 2) is True
    assert expr.participates("l", NL + NL2 // 2) is True  # d2 组行
    assert expr.participates("l", NL + NL2) is False       # 越界
    assert expr.participates("r", NR - 1) is True
    assert expr.participates("r", NR + NR2 + 100) is False
    stmts_used = store.conn.statements - stmts0
    print(f"6 次参与判断使用 SQL 语句: {stmts_used}（与 {expected_combos:,} 组合无关）")
    assert stmts_used < 200, "参与查询访问量异常，疑似展开全部组合"
    # 可直接按实例 + 序号审计来源行内容
    row0 = sess.source_row("l", 0)
    print("来源行 l#0:", row0)
    assert row0["values"] == {"dev": "d", "part": 7}

    # 不存在的元组 -> 空表达，节点为 0
    absent = sess.lineage((999,))
    assert not absent.exists() and absent.node_count() == 0
    assert absent.combination_count() == 0
    print("不存在元组 (999,) 的表达节点数:", absent.node_count())

    print()
    print("=" * 72)
    print("阶段四：新进程重开，无内存 Spec / 输入文件仍可追溯")
    print("=" * 72)
    code = (
        "import sys; sys.path.insert(0, %r);"
        "from factor_delivery import DeliveryStore, QuerySession;"
        "s = DeliveryStore.open(%r); q = QuerySession(s);"
        "e = q.lineage((7,));"
        "assert e.exists();"
        "assert e.combination_count() == %d;"
        "assert e.node_count() <= %d;"
        "assert e.participates('l', 123) is True;"
        "assert e.participates('l', %d) is False;"
        "assert q.lineage((999,)).exists() is False;"
        "print('CHILD_OK', e.combination_count(), e.node_count())"
    ) % (
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        path, expected_combos, source_rows + 50, NL + NL2,
    )
    proc = subprocess.run(
        [sys.executable, "-I", "-c", code],
        capture_output=True, text=True,
    )
    if proc.returncode != 0:
        print(proc.stderr)
        raise AssertionError("子进程重开追溯失败")
    print("子进程输出:", proc.stdout.strip())
    store.close()

    print()
    print("=" * 72)
    print("阶段五：小规模独立枚举器核对来源组合与重数")
    print("=" * 72)
    # 同结构小样本（数值不同以保留多个输出元组），逐组合枚举比对
    small = _small_analogue()
    spath = os.path.join(tmp, "small.db")
    sstore = DeliveryStore.create(spath, small)
    build_factorized(sstore, small)
    sstore.close()
    ssess = QuerySession(DeliveryStore.open(spath))
    enum = brute_force_lineage(small)
    # 复用测试里的独立展开器
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from test_lineage import expand_expression
    for tup, combos in enum.items():
        e = ssess.lineage(tup)
        assert e.combination_count() == sum(combos.values())
        assert expand_expression(e) == combos, tup
        print(f"元组 {tup}: 组合数 {e.combination_count()}，"
              f"{len(combos)} 个来源组合全部匹配")
    sstore_ref = ssess.store
    sstore_ref.close()
    print()
    print("全部规模化演练断言通过。")


def _small_analogue():
    """4 设备部件 × 3 供应点的小样本，含 multiplicity 与同值异序行。"""
    device = Table(
        [Column("id", "text", False, "device")],
        [("d",), ("d2",)], multiplicities=[2, 1],
    )
    left = Table(
        [Column("dev", "text", False, "device"),
         Column("part", "int64", False, "part")],
        [("d", 7), ("d", 7), ("d2", 7), ("d2", 8)],  # 前两行同值异序
    )
    right = Table(
        [Column("dev", "text", False, "device"),
         Column("sup", "int64", False, "supply")],
        [("d", 9), ("d", 9), ("d2", 9)],
    )
    return Spec(
        tables={"device": device, "left": left, "right": right},
        instances={"d": "device", "l": "left", "r": "right"},
        edges=[Edge("d", ("id",), "l", ("dev",)),
               Edge("d", ("id",), "r", ("dev",))],
        output_attrs=["part"],
        name="lineage_small",
    )


if __name__ == "__main__":
    main()
