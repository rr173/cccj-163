"""可运行的批次更新调用示例：演示批次编号与记录身份的生命周期。

场景：设备目录的部件表（left）与供货点表（right）按周修订。
本示例依次演示：

  1. 首次全量构建交付文件；
  2. 批次 B-2026-W47：两端同时插入（交互增量）+ 一条重复次数修正；
  3. 批次编号去重：同号同内容重放返回原结果，同号异内容被明确拒绝；
  4. 同内容不同身份：插入一条与旧记录值完全相同的行，得到全新 rid/row_seq；
  5. 按旧 rid 撤销；
  6. 关闭后在新进程重开交付文件（无需原始数据集），版本与批次去重信息齐全。

运行：python3 -I examples/batch_update_lifecycle.py
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from factor_delivery import (
    Column, Table, Edge, Spec,
    build_factorized, DeliveryStore, QuerySession,
    ChangeSet, BatchChange, BatchConflictError, apply_batch,
)


def build_initial_spec():
    device = Table(
        [Column("id", "text", False, "device")],
        [("d-100",), ("d-200",)],
    )
    parts = Table(
        [Column("dev", "text", False, "device"),
         Column("part", "text", False, "part")],
        [("d-100", "轴承"), ("d-100", "齿轮"), ("d-200", "外壳")],
    )
    supplies = Table(
        [Column("dev", "text", False, "device"),
         Column("sup", "text", False, "supply")],
        [("d-100", "华北仓"), ("d-200", "华南仓")],
    )
    return Spec(
        tables={"device": device, "parts": parts, "supplies": supplies},
        instances={"d": "device", "l": "parts", "r": "supplies"},
        edges=[Edge("d", ("id",), "l", ("dev",)),
               Edge("d", ("id",), "r", ("dev",))],
        output_attrs=["device", "part", "supply"],
        name="device_catalog",
    )


def main():
    workdir = tempfile.mkdtemp()
    path = os.path.join(workdir, "catalog.db")

    # ---------- 0. 首次全量构建（一次性输入） ----------
    spec = build_initial_spec()
    store = DeliveryStore.create(path, spec)
    build_factorized(store, spec)
    store.close()
    print(f"交付文件已建立：{path}")

    store = DeliveryStore.open(path)
    sess = QuerySession(store)
    print(f"初始版本 total = {sess.count()}")

    # ---------- 1. 批次：两端同时插入 + 一条 mult 修正 ----------
    # rid 由系统分配；调用方不需要也不应该指定新记录身份。
    cs = ChangeSet("B-2026-W47")
    cs.add(BatchChange.insert("parts", ("d-200", "密封圈"), mult=1))
    cs.add(BatchChange.insert("supplies", ("d-200", "西南仓"), mult=1))
    cs.add(BatchChange.update_mult("parts", 0, new_mult=4))  # 轴承 ×4
    result = apply_batch(store, cs)
    print("\n[B-2026-W47] 提交成功")
    print(f"  版本号 applied_seq = {result.applied_seq}")
    print(f"  新 total = {result.total_weight}")
    print(f"  新分配 rid（按批次内位置）= {result.inserted_rid}")
    print(f"  重建分区 {result.rebuilt_partitions}，"
          f"种子化复用分区 {result.seeded_partitions}")
    new_parts_rid = result.inserted_rid["0"][1]
    new_sup_rid = result.inserted_rid["1"][1]

    # ---------- 2. 批次编号去重 ----------
    # 同号同内容：返回原处理结果，不产生新版本
    replay = apply_batch(store, ChangeSet("B-2026-W47", list(cs.changes)))
    print(f"\n重放同号同内容：applied_seq 仍为 {replay.applied_seq}，"
          f"total 仍为 {replay.total_weight}")
    # 同号异内容：明确拒绝
    try:
        apply_batch(store, ChangeSet("B-2026-W47", [
            BatchChange.insert("parts", ("d-100", "垫片"), 1),
        ]))
        raise AssertionError("编号冲突应当被拒绝")
    except BatchConflictError as exc:
        print(f"同号异内容被拒绝：{type(exc).__name__}")

    # ---------- 3. 同内容不同身份 ----------
    # 插入一条与 rid=0 值完全相同的部件行；新行获得全新 rid 与 row_seq，
    # 绝不复用旧身份，即使内容逐字节相同。
    same = apply_batch(store, ChangeSet("B-2026-W48a", [
        BatchChange.insert("parts", ("d-100", "轴承"), mult=1),
    ]))
    duplicate_rid = same.inserted_rid["0"][1]
    print(f"\n同值新插入：旧 rid=0 不复用，新记录 rid={duplicate_rid}")
    row_old = store.fetch_source_row("l", 0)
    row_new = store.fetch_source_row("l", duplicate_rid)
    print(f"  旧行 row_seq=0 值={row_old['values']} mult={row_old['mult']}")
    print(f"  新行 row_seq={row_new['row_seq']} 值={row_new['values']} "
          f"mult={row_new['mult']}（身份独立）")

    # ---------- 4. 按旧身份撤销 ----------
    # 撤销只接受存活的旧 rid；撤销后该身份不再出现在任何来源表达中。
    apply_batch(store, ChangeSet("B-2026-W48b", [
        BatchChange.retract("parts", new_parts_rid),
        BatchChange.retract("supplies", new_sup_rid),
    ]))
    print(f"\n已撤销本批次早先插入的 rid={new_parts_rid}/{new_sup_rid}")
    print(f"撤销后 parts 来源行数 = {store.source_row_count('l')}")
    # 撤销后审计该 row_seq 返回 None（身份保留为历史，但不再存活）
    assert store.fetch_source_row("l", new_parts_rid) is None

    # ---------- 5. 批次日志可审计 ----------
    print("\n已登记批次（按应用顺序）：")
    for rec in store.list_batches():
        print(f"  seq={rec['applied_seq']} {rec['batch_no']} "
              f"hash={rec['content_hash'][:12]}…")
    store.close()

    # ---------- 6. 新进程重开：无需最初数据集 ----------
    store = DeliveryStore.open(path)
    sess = QuerySession(store)
    print(f"\n重开交付文件：当前 total = {sess.count()}，"
          f"已知批次 {[b['batch_no'] for b in store.list_batches()]}")
    # 批次去重信息仍然有效
    again = apply_batch(store, ChangeSet("B-2026-W47", list(cs.changes)))
    print(f"重开后重放 B-2026-W47 仍命中原结果 seq={again.applied_seq}")
    print("\n示例运行完成。")
    store.close()


if __name__ == "__main__":
    main()
