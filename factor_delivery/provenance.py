"""来源表达（lineage / provenance）的构建期节点。

来源表达回答「这条输出元组由哪些输入行组合产生」。它是一棵随 *来源行与
因子结构* 增长、而不随来源组合数增长的紧凑 DAG，三种节点：

PSrc  某个表实例（角色别名）上的一组来源行。每行只登记
      (装载序号 row_seq, multiplicity)，mult 是该行代表的重复次数——
      绝不按 mult 逐份制造来源节点。weight = 各行 mult 之和。
PProd 连接分支相乘：同一组合中各实例的来源共同贡献，weight = 各分支之积。
PAlt  投影后的袋并：多个连接分支都能得到同一输出元组，weight = 各分支之和。
      不同连接键分组的来源只在此相加，永不重新交叉相乘。

语义不变量（与因子 DAG 同构）：

* 任一非空因子节点的来源闭包 weight 等于该节点的 factor weight；
* 一条输出元组的来源表达，其 weight 恰好等于该元组在结果袋中的重数；
* 来源组合 = 从表达根到各 PSrc 叶子选一行的所有选法（PAlt 取其一、
  PProd 全部取），数量由 weight 直接给出，不需要枚举。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple


class ProvenanceError(RuntimeError):
    """来源表达构建/读取失败。"""


@dataclass
class PNode:
    weight: int
    kind: str = "?"
    prov_id: Optional[int] = None  # 持久化后分配的 id


@dataclass
class PSrc(PNode):
    """一个表实例上的来源行集合：alias -> [(row_seq, mult), ...]。

    rows 按 row_seq 升序、同一 row_seq 只出现一次（合并 mult）。
    """

    alias: str = ""
    rows: List[Tuple[int, int]] = field(default_factory=list)

    def __init__(self, alias: str, rows: Sequence[Tuple[int, int]]):
        merged = _merge_rows(rows)
        if not merged:
            raise ProvenanceError("src 来源节点至少需要一条来源行")
        super().__init__(weight=sum(m for _, m in merged), kind="src")
        self.alias = alias
        self.rows = merged


@dataclass
class PProd(PNode):
    """连接相乘：各分支在同一组合中共同贡献。"""

    children: List[PNode] = field(default_factory=list)

    def __init__(self, children: Sequence[PNode]):
        if len(children) < 2:
            raise ProvenanceError("prod 来源节点至少需要两个分支")
        w = 1
        for c in children:
            w *= c.weight
        super().__init__(weight=w, kind="prod")
        self.children = list(children)


@dataclass
class PAlt(PNode):
    """袋并相加：互斥的多个来源分支都能得到同一输出元组。"""

    children: List[PNode] = field(default_factory=list)

    def __init__(self, children: Sequence[PNode]):
        if len(children) < 2:
            raise ProvenanceError("alt 来源节点至少需要两个分支")
        super().__init__(weight=sum(c.weight for c in children), kind="alt")
        self.children = list(children)


def _merge_rows(rows: Sequence[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """合并同一 row_seq 的 multiplicity；完全相同的值若序号不同仍是不同行，
    因此这里只按 *序号* 合并，绝不按值合并。"""
    acc: Dict[int, int] = {}
    for seq, mult in rows:
        if not isinstance(seq, int) or isinstance(seq, bool) or seq < 0:
            raise ProvenanceError(f"非法来源行序号 {seq!r}")
        if not isinstance(mult, int) or isinstance(mult, bool) or mult <= 0:
            raise ProvenanceError(f"来源行 {seq} 的 multiplicity 非法: {mult!r}")
        acc[seq] = acc.get(seq, 0) + mult
    return sorted(acc.items())


# --------------------------------------------------------------------- #
# 规范化构造：展平同类复合节点、合并同实例 src、结构驻留（interning）
# --------------------------------------------------------------------- #
class ProvFactory:
    """按结构驻留来源节点：同构来源表达只持久化一份，节点数随结构增长。"""

    def __init__(self) -> None:
        self.src_intern: Dict[Tuple[str, Tuple[Tuple[int, int], ...]], PSrc] = {}
        self.prod_intern: Dict[Tuple[Any, ...], PProd] = {}
        self.alt_intern: Dict[Tuple[Any, ...], PAlt] = {}

    def src(self, alias: str, rows: Sequence[Tuple[int, int]]) -> PSrc:
        merged = tuple(_merge_rows(rows))
        key = (alias, merged)
        node = self.src_intern.get(key)
        if node is None:
            node = PSrc(alias, merged)
            self.src_intern[key] = node
        return node

    def prod(self, children: Sequence[PNode]) -> PNode:
        flat: List[PNode] = []
        for c in children:
            # 乘法结合律：嵌套 prod 展平
            flat.extend(c.children if c.kind == "prod" else [c])
        if len(flat) == 1:
            return flat[0]
        key = tuple(id(c) for c in flat)
        node = self.prod_intern.get(key)
        if node is None:
            node = PProd(flat)
            self.prod_intern[key] = node
        return node

    def alt(self, children: Sequence[PNode]) -> PNode:
        flat: List[PNode] = []
        for c in children:
            # 加法结合律：嵌套 alt 展平
            flat.extend(c.children if c.kind == "alt" else [c])
        if len(flat) == 1:
            return flat[0]
        # 同实例的 src 分支合并 mult（袋语义相加），保持节点紧凑
        merged: List[PNode] = []
        src_buckets: Dict[str, List[Tuple[int, int]]] = {}
        for c in flat:
            if c.kind == "src":
                src_buckets.setdefault(c.alias, []).extend(c.rows)
            else:
                merged.append(c)
        for alias, rows in src_buckets.items():
            merged.append(self.src(alias, rows))
        if len(merged) == 1:
            return merged[0]
        key = tuple(id(c) for c in merged)
        node = self.alt_intern.get(key)
        if node is None:
            node = PAlt(merged)
            self.alt_intern[key] = node
        return node


@dataclass
class PLazyRef(PNode):
    """对已持久化来源节点的惰性引用（增量更新种子化干净分区时使用）。

    它不再展开/复用旧 PNode 对象，只记住磁盘 prov_id 与 weight；
    持久化阶段遇到它直接写回同一个 prov_id，旧来源身份（含未改记录的
    row_seq）因此原样保留。被引用节点在新版本不可达时由清理逻辑回收。
    """

    def __init__(self, prov_id: int, weight: int):
        super().__init__(weight=weight, kind="lazy_ref", prov_id=prov_id)
