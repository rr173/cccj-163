"""查询层：重开交付文件后仅靠因子节点表工作。

运行时因子是惰性加载的 DAG，每个节点的重数都从子节点自底向上重算并与
fd_node.weight 缓存值交叉核对，查询从不只信一个总行数字段。

秩寻址（从 0 开始，重复元组按 multiplicity 各占一秩）：

* leaf   —— 整个节点只有一袋重复的同一元组，任意合法秩都返回该元组；
* group  —— 按全序跳过前缀 entry（累计子权重），落到某 entry 后递归；
* product —— 混合进制：按分支顺序，q_i = (r // stride_i) mod weight_i，
             各分支独立寻址，绝不把不同父键下的候选相乘。

条件读取只支持输出列上的等值或 is_null；筛选后构建受限因子并重新计算
连续秩，而不是从未筛选的 r 处截取。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .store import DeliveryStore, ProtocolError


class QueryError(ValueError):
    """查询参数非法（未知列、类型不符、负秩等）。"""


def _match_condition(session: "QuerySession", attr: str,
                     cond: "Condition", value: Any) -> bool:
    """按属性声明类型严格匹配，规避 True==1 / False==0 的 Python 陷阱。"""
    if cond.op == "is_null":
        return value is None
    if value is None:
        return False
    dtype = session.dtype_of(attr)
    if dtype == "bool":
        # 磁盘读回的 bool 已还原为 Python bool；仍按真值严格判等
        return isinstance(value, bool) and value == cond.value
    if dtype == "int64":
        # int64 列绝不能把 bool 当作匹配值
        return isinstance(value, int) and not isinstance(value, bool) \
            and value == cond.value
    return value == cond.value


@dataclass(frozen=True)
class Condition:
    attr: str
    op: str            # "eq" | "is_null"
    value: Any = None  # eq 时的非 None 值

    @staticmethod
    def eq(attr: str, value: Any) -> "Condition":
        if value is None:
            raise QueryError("等值条件不接受 None，请使用 is_null")
        return Condition(attr, "eq", value)

    @staticmethod
    def is_null(attr: str) -> "Condition":
        return Condition(attr, "is_null", None)


# --------------------------------------------------------------------- #
# 运行时因子（惰性）
# --------------------------------------------------------------------- #
def _value_identity(value: Any) -> Tuple[str, Any]:
    """与构建期一致：区分 bool 与 int，None 单独成类。"""
    if value is None:
        return ("null", None)
    if isinstance(value, bool):
        return ("bool", value)
    return (type(value).__name__, value)


def _coalesce_nodes(session: "QuerySession", branches: List["RNode"],
                    scope: Tuple[str, ...]) -> "RNode":
    """把同作用域的多个袋并分支归并成按全序排列的单个等价因子。

    递归规则：
      leaf    -> 重数相加为一个 leaf；
      group   -> 按取值合并 entry，同值子因子再归并（袋并，重数相加）；
      product -> 同位置分支各自归并，重新条件独立乘积；
      union   -> 展平后一并归并。
    只遍历因子节点与分组 entry，数量随源表/分组规模而非逻辑输出规模。
    """
    # 展平嵌套 union、剔除空分支
    flat: List[RNode] = []
    stack = list(branches)
    while stack:
        n = stack.pop()
        n._ensure()
        if n.kind == "empty":
            continue
        if n.kind == "union":
            stack.extend(_product_children(session, n))
        else:
            flat.append(n)
    if not flat:
        return REmpty(session, scope)
    if len(flat) == 1:
        return flat[0]

    kinds = {n.kind for n in flat}
    if kinds == {"leaf"}:
        return _VirtualLeaf(session, scope, sum(n.weight for n in flat))
    if kinds == {"group"}:
        attrs = {n._group_attr() for n in flat}
        if len(attrs) != 1:
            raise ProtocolError("袋并 group 的分组属性不一致")
        attr = next(iter(attrs))
        # 收集每个取值下的子因子。group 有两种表示：
        # 磁盘 RNode 的 _entries 为 (value, child_id)，
        # 归并产物 RGroupVirtual 的 _virtual_entries 为 (value, child RNode)。
        buckets: Dict[Tuple[str, Any], List[RNode]] = {}
        value_of: Dict[Tuple[str, Any], Any] = {}
        for gn in flat:
            for value, child in _iter_group_entries(session, gn):
                k = _value_identity(value)
                buckets.setdefault(k, []).append(child)
                value_of[k] = value
        sub_scope = tuple(a for a in scope if a != attr)
        entries: List[Tuple[Any, RNode]] = []
        for k, subs in buckets.items():
            entries.append((value_of[k],
                            _coalesce_nodes(session, subs, sub_scope)))
        entries.sort(key=lambda item: session.store_value_key(attr, item[0]))
        return RGroupVirtual(session, scope, attr, entries)
    if kinds == {"product"}:
        # 单分支 product 退化为其唯一孩子，避免无意义的乘积包装
        unwrapped = []
        for n in flat:
            ch = _product_children(session, n)
            unwrapped.append(ch[0] if len(ch) == 1 else n)
        if any(n.kind != "product" for n in unwrapped):
            return _coalesce_nodes(session, unwrapped, scope)
        # 关键：(a×b) ∪ (c×d) ≠ (a∪c)×(b∪d)。袋并不能下沉到 product
        # 内部按分支位置重新乘积（restrict 下推时各分支存活情况不同，
        # 重乘会凭空造出跨父键组合）。保留每个 product 为独立袋并分支，
        # 构造虚拟 union，寻址时按全序做带局部秩映射的多路归并。
        return _VirtualUnion(session, scope, unwrapped)
    # group/product 异构（理论上规范化后不应出现）
    raise ProtocolError(f"无法归并异构袋并分支: {kinds}")


def _iter_group_entries(session: "QuerySession",
                        gn: "RNode") -> List[Tuple[Any, "RNode"]]:
    """统一遍历 group 的 (值, 子因子)，兼容磁盘节点与虚拟节点。"""
    virtual = getattr(gn, "_virtual_entries", None)
    if virtual is not None:
        return list(virtual)
    gn._ensure()
    if gn._entries is None:
        return []
    return [(value, session.get_node(cid))
            for value, cid in gn._entries]


def _product_children(session: "QuerySession",
                      n: "RNode") -> List["RNode"]:
    """统一获取 product 的子因子，兼容磁盘节点与 RProductVirtual。"""
    virtual = getattr(n, "_vchildren", None)
    if virtual is not None:
        return list(virtual)
    n._ensure()
    return [session.get_node(cid) for cid in n._child_ids]  # type: ignore[union-attr]


def _split_first_column(session: "QuerySession",
                        n: "RNode") -> List[Tuple[Any, "RNode"]]:
    """把因子按输出首列拆成 (首列值, 剩余作用域因子) 的枚举。

    group：直接取其 entry；
    product：首分支承载输出首列，逐 entry 把剩余分支重新乘积。
    """
    if n.kind == "group":
        return _iter_group_entries(session, n)
    if n.kind == "product":
        children = [c for c in _product_children(session, n) if c.scope]
        if len(children) == 1:
            # 其余都是空作用域 multiplicity 叶子，首列即唯一有作用域分支
            return _split_first_column(session, children[0])
        first, rest = children[0], children[1:]
        out: List[Tuple[Any, RNode]] = []
        for value, sub in _iter_group_entries(session, first):
            if not rest:
                tail: RNode = sub
            elif len(rest) == 1:
                tail = _recombine_product(session, [sub, rest[0]])
            else:
                tail = _recombine_product(session, [sub] + rest)
            out.append((value, tail))
        return out
    if n.kind == "union":
        # 展平一层虚拟袋并
        out = []
        for br in _product_children(session, n):
            out.extend(_split_first_column(session, br))
        return out
    raise ProtocolError(f"无法按首列拆分节点类型 {n.kind}")


def _recombine_product(session: "QuerySession",
                       children: List["RNode"]) -> "RNode":
    if len(children) == 1:
        return children[0]
    scope: List[str] = []
    for c in children:
        scope.extend(c.scope)
    return RProductVirtual(session, tuple(scope), children)


class RNode:
    def __init__(self, session: "QuerySession", node_id: int):
        self.session = session
        self.node_id = node_id
        cur = session.conn.execute(
            "SELECT kind, weight, scope FROM fd_node WHERE node_id=?",
            (node_id,),
        )
        row = cur.fetchone()
        if row is None:
            raise ProtocolError(f"因子节点 {node_id} 缺失")
        self.kind, cached, scope_text = row
        self.cached_weight = int(cached)
        self.scope: Tuple[str, ...] = tuple(
            s for s in scope_text.split(",") if s
        )
        self._loaded = False
        self._entries: Optional[List[Tuple[Any, int]]] = None  # (value, child_id)
        self._child_ids: Optional[List[int]] = None
        self._computed_weight: Optional[int] = None

    def _ensure(self) -> None:
        if self._loaded:
            return
        if self.kind == "group":
            cur = self.session.conn.execute(
                "SELECT is_null, v_int, v_text, v_bool, child_id, attr "
                "FROM fd_group_entry WHERE node_id=? ORDER BY ord",
                (self.node_id,),
            )
            entries = []
            for is_null, v_int, v_text, v_bool, cid, attr in cur.fetchall():
                if is_null:
                    value: Any = None
                else:
                    dtype = self.session.dtype_of(attr)
                    # 注意 bool 是 int 的子类，必须先按 dtype 选列，
                    # 不能用 isinstance 顺序猜测
                    if dtype == "bool":
                        value = bool(v_bool)
                    elif dtype == "int64":
                        value = v_int
                    else:
                        value = v_text
                entries.append((value, cid))
            self._entries = entries
        elif self.kind in ("product", "union"):
            cur = self.session.conn.execute(
                "SELECT child_id FROM fd_product_child "
                "WHERE node_id=? ORDER BY position",
                (self.node_id,),
            )
            self._child_ids = [r[0] for r in cur.fetchall()]
        self._loaded = True

    @property
    def weight(self) -> int:
        """自底向上重算的重数，并核对缓存值（受限/虚拟节点无磁盘缓存，免核对）。"""
        if self._computed_weight is None:
            self._computed_weight = self._recompute()
            if self.node_id > 0 and self.cached_weight >= 0:
                if self._computed_weight != self.cached_weight:
                    raise ProtocolError(
                        f"节点 {self.node_id} 缓存重数 {self.cached_weight} "
                        f"与按子节点重算值 {self._computed_weight} 不符"
                    )
        return self._computed_weight

    def _recompute(self) -> int:
        self._ensure()
        if self.kind == "empty":
            return 0
        if self.kind == "leaf":
            return self.cached_weight  # 叶子重数源自源 multiplicity 乘积/合并
        if self.kind == "group":
            total = 0
            for _, cid in self._entries:  # type: ignore[union-attr]
                total += self.session.get_node(cid).weight
            return total
        if self.kind == "product":
            total = 1
            for cid in self._child_ids:  # type: ignore[union-attr]
                total *= self.session.get_node(cid).weight
            return total
        if self.kind == "union":
            # 袋并：各分支重数相加
            return sum(
                self.session.get_node(cid).weight
                for cid in self._child_ids  # type: ignore[union-attr]
            )
        raise ProtocolError(f"未知节点类型 {self.kind}")

    # ---------------- 秩寻址 ---------------- #
    def row_at(self, r: int, out: List[Any]) -> None:
        self._ensure()
        if self.kind == "empty":
            raise QueryError("空结果上没有可寻址的秩")
        if self.kind == "leaf":
            return  # 叶子不产生输出列，值已由外层 group 写入
        if self.kind == "group":
            skipped = 0
            for value, cid in self._entries:  # type: ignore[union-attr]
                child = self.session.get_node(cid)
                w = child.weight
                if r < skipped + w:
                    out.append(value)
                    child.row_at(r - skipped, out)
                    return
                skipped += w
            raise QueryError(f"秩 {r} 超出 group 节点范围")
        if self.kind == "product":
            # 条件独立分支的笛卡尔积，按分支（输出列）顺序定全序。
            # 不能假设各分支等宽：group 分支可能把来源行 multiplicity
            # 折叠进 entry 叶子，使「不同取值数」与分支 weight 不等。
            # 改用 entry 宽度感知寻址：从最右分支起，每条左分支 entry 的
            # 步长 = 该 entry 子权重 × 右侧全部兄弟的总重数。
            self._product_row_at(r, out)
            return
        if self.kind == "union":
            # 袋并：惰性归并为按全序等价的因子后寻址（见 coalesce）
            self.coalesce().row_at(r, out)
            return
        raise ProtocolError(f"未知节点类型 {self.kind}")

    def _product_row_at(self, r: int, out: List[Any]) -> None:
        """entry 宽度感知的 product 秩寻址，与参考袋的全序严格一致。

        全局展开等价于嵌套循环：沿输出列从左到右选 entry，每个完整取值
        组合再按其重数（沿途各 entry 子权重之积）连续重复。递归传递
        「左侧已选 entry 的子权重累积 left_mult」：

        * 当前 group 分支某 entry（子权重 w）的块宽 = w × left_mult × 右侧总重；
        * 进入 entry 后，右侧兄弟的 entry 切换周期变为 left_mult × w；
        * 最右分支定位到 entry 后，剩余偏移即该元组的 multiplicity 重复，
          leaf 不产生输出列，重复不改变写值。
        """
        self._ensure()
        child_ids = self._child_ids  # type: ignore[assignment]
        self._address_product(0, r, 1, out)

    def _address_product(self, idx: int, r: int,
                         left_mult: int, out: List[Any]) -> None:
        child_ids = self._child_ids  # type: ignore[union-attr]
        node = self.session.get_node(child_ids[idx])
        last = idx == len(child_ids) - 1
        right_w = self._suffix_weight(idx + 1)
        if node.kind == "group":
            node._ensure()
            for value, cid in node._entries:  # type: ignore[union-attr]
                child = self.session.get_node(cid)
                block = child.weight * left_mult * right_w
                if r < block:
                    out.append(value)
                    new_left = left_mult * child.weight
                    # 块内偏移直接作为右侧分支视角的秩：右层 entry 的全局
                    # 块宽天然含 left_mult 周期，由右层自行按块定位。
                    within = r % (child.weight * right_w)
                    child.row_at(within // right_w, out)  # leaf，无输出列
                    if not last:
                        self._address_product(idx + 1, within, new_left, out)
                    return
                r -= block
            raise QueryError(f"product group 分支 {idx} 秩越界")
        # 非 group 分支（leaf / 嵌套 product）：整体一块
        block = node.weight * left_mult * right_w
        if r < block:
            new_left = left_mult * node.weight
            within = r % (node.weight * right_w)
            node.row_at(within // right_w, out)
            if not last:
                self._address_product(idx + 1, within, new_left, out)
            return
        raise QueryError(f"product 分支 {idx} 秩越界")

    def _suffix_weight(self, start: int) -> int:
        """从分支 start 起所有右侧兄弟的总重数之积（空区间为 1）。"""
        child_ids = self._child_ids  # type: ignore[union-attr]
        total = 1
        for j in range(start, len(child_ids)):
            total *= self.session.get_node(child_ids[j]).weight
        return total

    # ---------------- 袋并归并 ---------------- #
    def coalesce(self) -> "RNode":
        """把 union（同作用域多分支的袋并）惰性归并成按全序排列的等价因子。

        各分支可能在同一取值上重叠；归并时相同取值的子因子再做袋并，
        重数相加。结果被缓存，且只遍历 union 的分组结构，不展开逻辑输出。
        """
        self._ensure()
        cached = getattr(self, "_coalesced", None)
        if cached is not None:
            return cached
        branches = [self.session.get_node(cid) for cid in self._child_ids]  # type: ignore[arg-type]
        merged = _coalesce_nodes(self.session, branches, self.scope)
        self._coalesced = merged
        return merged

    # ---------------- 条件限制 ---------------- #
    def restrict(self, conditions: Dict[str, Condition]) -> "RNode":
        if not conditions:
            return self
        self._ensure()
        if self.kind in ("empty", "leaf"):
            return self
        if self.kind == "group":
            attr = self._group_attr()
            if attr in conditions:
                cond = conditions[attr]
                kept: List[Tuple[Any, int]] = []
                for value, cid in self._entries:  # type: ignore[union-attr]
                    if _match_condition(self.session, attr, cond, value):
                        kept.append((value, cid))
                if not kept:
                    return REmpty(self.session, self.scope)
                # 本层条件已消费，其余条件继续向子节点下推
                remaining = {a: c for a, c in conditions.items() if a != attr}
                if not remaining:
                    return RGroupRestricted(self.session, self, attr, kept)
                sub_entries: List[Tuple[Any, RNode]] = []
                for value, cid in kept:
                    sub = self.session.get_node(cid).restrict(remaining)
                    if sub.weight > 0:
                        sub_entries.append((value, sub))
                if not sub_entries:
                    return REmpty(self.session, self.scope)
                return RGroupVirtual(
                    self.session, self.scope, attr, sub_entries
                )
            # 条件落在更深层：逐 entry 限制
            kept_entries: List[Tuple[Any, RNode]] = []
            for value, cid in self._entries:  # type: ignore[union-attr]
                sub = self.session.get_node(cid).restrict(conditions)
                if sub.weight > 0:
                    kept_entries.append((value, sub))
            if not kept_entries:
                return REmpty(self.session, self.scope)
            return RGroupVirtual(self.session, self.scope, attr, kept_entries)
        if self.kind == "product":
            restricted_children = [
                self.session.get_node(cid).restrict(conditions)
                for cid in self._child_ids  # type: ignore[union-attr]
            ]
            if any(c.weight == 0 for c in restricted_children):
                return REmpty(self.session, self.scope)
            return RProductVirtual(self.session, self.scope, restricted_children)
        if self.kind == "union":
            # 对每个袋并分支独立筛选，再重新归并（重数在筛选语义上相加）
            restricted = [
                self.session.get_node(cid).restrict(conditions)
                for cid in self._child_ids  # type: ignore[union-attr]
            ]
            if all(c.weight == 0 for c in restricted):
                return REmpty(self.session, self.scope)
            return _coalesce_nodes(self.session, restricted, self.scope)
        raise ProtocolError(f"未知节点类型 {self.kind}")

    def _group_attr(self) -> str:
        # 虚拟 group（受限/归并产生，id=-1）在内存中带 attr，无需查盘
        va = getattr(self, "attr", None)
        if va is not None:
            return va
        cur = self.session.conn.execute(
            "SELECT attr FROM fd_group_entry WHERE node_id=? LIMIT 1",
            (self.node_id,),
        )
        row = cur.fetchone()
        if row is None:
            raise ProtocolError(f"group 节点 {self.node_id} 没有任何 entry")
        return row[0]


class REmpty(RNode):
    def __init__(self, session: "QuerySession", scope: Tuple[str, ...]):
        self.session = session
        self.node_id = -1
        self.kind = "empty"
        self.cached_weight = 0
        self.scope = scope
        self._loaded = True
        self._entries = None
        self._child_ids = None
        self._computed_weight = 0

    def _ensure(self) -> None:
        pass

    def weight_override(self) -> int:
        return 0


class RGroupRestricted(RNode):
    """条件正好作用在本 group 属性上：只保留磁盘 entry 的一个子集。"""

    def __init__(self, session: "QuerySession", base: RNode,
                 attr: str, kept: List[Tuple[Any, int]]):
        self.session = session
        self.node_id = -1  # 受限虚拟节点，没有独立磁盘缓存
        self.kind = "group"
        self.cached_weight = -1
        self.scope = base.scope
        self._loaded = True
        self.attr = attr
        self._entries = kept
        self._child_ids = None
        self._computed_weight = None
        self._base_node_id = base.node_id

    def _recompute(self) -> int:
        # 只按保留下来的 entry 重算，不能使用基础节点的缓存总重数
        return sum(
            self.session.get_node(cid).weight for _, cid in self._entries  # type: ignore[union-attr]
        )


class RGroupVirtual(RNode):
    """条件作用在更深层：entry 集合改变，子因子是受限后的虚拟节点。"""

    def __init__(self, session: "QuerySession", scope: Tuple[str, ...],
                 attr: str, entries: List[Tuple[Any, RNode]]):
        self.session = session
        self.node_id = -1
        self.kind = "group"
        self.cached_weight = -1
        self.scope = scope
        self._loaded = True
        self.attr = attr
        self._virtual_entries = entries
        self._entries = None
        self._child_ids = None
        self._computed_weight = None

    def _ensure(self) -> None:
        pass

    def _recompute(self) -> int:
        return sum(child.weight for _, child in self._virtual_entries)

    def row_at(self, r: int, out: List[Any]) -> None:
        skipped = 0
        for value, child in self._virtual_entries:
            w = child.weight
            if r < skipped + w:
                out.append(value)
                child.row_at(r - skipped, out)
                return
            skipped += w
        raise QueryError(f"秩 {r} 超出虚拟 group 范围")

    def restrict(self, conditions: Dict[str, Condition]) -> RNode:
        remaining = {a: c for a, c in conditions.items()}
        if self.attr in remaining:
            cond = remaining.pop(self.attr)
            kept: List[Tuple[Any, RNode]] = []
            for value, child in self._virtual_entries:
                if _match_condition(self.session, self.attr, cond, value):
                    kept.append((value, child))
            if not kept:
                return REmpty(self.session, self.scope)
            if not remaining:
                return RGroupVirtual(self.session, self.scope, self.attr, kept)
            return RGroupVirtual(
                self.session, self.scope, self.attr,
                [(v, c.restrict(remaining)) for v, c in kept],
            )._drop_empty()
        kept_entries = []
        for value, child in self._virtual_entries:
            sub = child.restrict(remaining)
            if sub.weight > 0:
                kept_entries.append((value, sub))
        if not kept_entries:
            return REmpty(self.session, self.scope)
        return RGroupVirtual(self.session, self.scope, self.attr, kept_entries)

    def _drop_empty(self) -> RNode:
        kept = [(v, c) for v, c in self._virtual_entries if c.weight > 0]
        if not kept:
            return REmpty(self.session, self.scope)
        self._virtual_entries = kept
        return self


class RProductVirtual(RNode):
    def __init__(self, session: "QuerySession", scope: Tuple[str, ...],
                 children: List[RNode]):
        self.session = session
        self.node_id = -1
        self.kind = "product"
        self.cached_weight = -1
        self.scope = scope
        self._loaded = True
        self._vchildren = children
        self._entries = None
        self._child_ids = None
        self._computed_weight = None

    def _ensure(self) -> None:
        pass

    def _recompute(self) -> int:
        total = 1
        for c in self._vchildren:
            total *= c.weight
        return total

    def row_at(self, r: int, out: List[Any]) -> None:
        # 与磁盘 product 同一套 entry 宽度感知寻址（group 分支可能折叠了
        # 来源 multiplicity，不能用规整混合进制）。递归传递左侧累积 mult。
        children = self._vchildren

        def suffix_w(start: int) -> int:
            total = 1
            for j in range(start, len(children)):
                total *= children[j].weight
            return total

        def address(idx: int, rank: int, left_mult: int) -> None:
            node = children[idx]
            last = idx == len(children) - 1
            right_w = suffix_w(idx + 1)
            entries = _iter_group_entries(self.session, node) \
                if node.kind == "group" else None
            if entries is not None:
                for value, child in entries:
                    block = child.weight * left_mult * right_w
                    if rank < block:
                        out.append(value)
                        new_left = left_mult * child.weight
                        within = rank % (child.weight * right_w)
                        child.row_at(within // right_w, out)
                        if not last:
                            address(idx + 1, within, new_left)
                        return
                    rank -= block
                raise QueryError("虚拟 product group 分支秩越界")
            block = node.weight * left_mult * right_w
            if rank < block:
                new_left = left_mult * node.weight
                within = rank % (node.weight * right_w)
                node.row_at(within // right_w, out)
                if not last:
                    address(idx + 1, within, new_left)
                return
            raise QueryError("虚拟 product 分支秩越界")

        address(0, r, 1)

    def restrict(self, conditions: Dict[str, Condition]) -> RNode:
        if not conditions:
            return self
        children = [c.restrict(conditions) for c in self._vchildren]
        if any(c.weight == 0 for c in children):
            return REmpty(self.session, self.scope)
        return RProductVirtual(self.session, self.scope, children)


class _VirtualLeaf(RNode):
    """归并产生的纯重数叶子（空作用域）。"""

    def __init__(self, session: "QuerySession", scope: Tuple[str, ...],
                 weight: int):
        self.session = session
        self.node_id = -1
        self.kind = "leaf"
        self.cached_weight = -1
        self.scope = scope
        self._loaded = True
        self._entries = None
        self._child_ids = None
        self._computed_weight = weight

    def _ensure(self) -> None:
        pass

    def row_at(self, r: int, out: List[Any]) -> None:
        return


class _VirtualUnion(RNode):
    """袋并的虚拟节点：保留各分支独立，寻址时按全序多路归并。

    每个分支本身已按输出全序排列。row_at 不展开整个输出，而是用「逐前缀
    列计数」在分支间定位：对最外层分组属性，统计每个分支在各取值上的重数，
    合并成全局有序取值域后落到对应分支的局部秩。
    """

    def __init__(self, session: "QuerySession", scope: Tuple[str, ...],
                 branches: List[RNode]):
        self.session = session
        self.node_id = -1
        self.kind = "union"
        self.cached_weight = -1
        self.scope = scope
        self._loaded = True
        self._vbranches = branches
        self._entries = None
        self._child_ids = None
        self._computed_weight = None

    def _ensure(self) -> None:
        pass

    def _recompute(self) -> int:
        return sum(b.weight for b in self._vbranches)

    def row_at(self, r: int, out: List[Any]) -> None:
        # 按输出首列把各分支拆成 (首列值, 剩余因子)，同值归入一个桶。
        # 桶内保留 *每个分支各自的剩余因子*，全局秩按分支顺序在桶内定位，
        # 命中后换算成该分支的局部秩再寻址——绝不把不同分支的剩余因子
        # 合并成一个乘积，避免跨父键交叉。
        attr = self.scope[0]
        value_order: List[Any] = []
        buckets: Dict[Tuple[str, Any], List[RNode]] = {}
        for br in self._vbranches:
            for value, tail in _split_first_column(self.session, br):
                k = _value_identity(value)
                if k not in buckets:
                    buckets[k] = []
                    value_order.append(value)
                buckets[k].append(tail)
        value_order.sort(
            key=lambda v: self.session.store_value_key(attr, v)
        )
        skipped = 0
        for value in value_order:
            tails = buckets[_value_identity(value)]
            bucket_w = sum(t.weight for t in tails)
            if r < skipped + bucket_w:
                out.append(value)
                local = r - skipped
                for tail in tails:
                    if local < tail.weight:
                        tail.row_at(local, out)
                        return
                    local -= tail.weight
                raise QueryError("虚拟袋并桶内局部秩越界")
            skipped += bucket_w
        raise QueryError(f"秩 {r} 超出虚拟袋并范围")

    def restrict(self, conditions: Dict[str, Condition]) -> RNode:
        restricted = [b.restrict(conditions) for b in self._vbranches]
        if all(b.weight == 0 for b in restricted):
            return REmpty(self.session, self.scope)
        return _coalesce_nodes(self.session, restricted, self.scope)


# --------------------------------------------------------------------- #
# 会话
# --------------------------------------------------------------------- #
class QuerySession:
    def __init__(self, store: DeliveryStore):
        self.store = store
        self.conn = store.conn
        self._cache: Dict[int, RNode] = {}
        self.output_attrs: List[str] = [a for a, _, _ in store.output_attrs]
        self._dtype = {a: t for a, t, _ in store.output_attrs}
        self._nullable = {a: n for a, _, n in store.output_attrs}
        cur = self.conn.execute(
            "SELECT value FROM fd_meta WHERE key='root_node_id'"
        )
        row = cur.fetchone()
        if row is None:
            raise ProtocolError("缺少根节点元数据")
        self.root_id = int(row[0])
        self.root = self.get_node(self.root_id)
        if self.root.kind != "empty" and self.root.scope != tuple(self.output_attrs):
            raise ProtocolError(
                f"根节点作用域 {self.root.scope} 与输出列 {self.output_attrs} 不符"
            )

    def get_node(self, node_id: int) -> RNode:
        if node_id < 0:
            raise ProtocolError(f"非法节点 id {node_id}")
        node = self._cache.get(node_id)
        if node is None:
            node = RNode(self, node_id)
            self._cache[node_id] = node
        return node

    def dtype_of(self, attr: str) -> str:
        return self._dtype[attr]

    def store_value_key(self, attr: str, value: Any) -> Tuple[Any, ...]:
        """全序键：NULL 最先；按属性 *声明类型* 分派（bool 经 SQLite 后是 int）。"""
        if value is None:
            return (0, 0, 0)
        if self._dtype[attr] == "bool":
            return (1, 0 if not value else 1, 0)
        return (1, 0, value)

    # ---------------- 公共 API ---------------- #
    def count(self) -> int:
        """精确总重数（触发全树自底向上重算与缓存核对）。"""
        return self.root.weight

    def page(self, r: int, n: int) -> List[Tuple[Any, ...]]:
        """全序下从秩 r 开始取 n 行；r 越界返回空，n=0 不枚举。"""
        if not isinstance(r, int) or isinstance(r, bool) or r < 0:
            raise QueryError("r 必须是非负整数")
        if not isinstance(n, int) or isinstance(n, bool) or n < 0:
            raise QueryError("n 必须是非负整数")
        total = self.count()
        if n == 0 or r >= total:
            return []
        end = min(r + n, total)
        return [self.row_at(i) for i in range(r, end)]

    def row_at(self, r: int) -> Tuple[Any, ...]:
        if not isinstance(r, int) or isinstance(r, bool) or r < 0:
            raise QueryError("r 必须是非负整数")
        out: List[Any] = []
        self.root.row_at(r, out)
        if len(out) != len(self.output_attrs):
            raise ProtocolError(
                f"寻址得到 {len(out)} 个值，输出列有 {len(self.output_attrs)} 个"
            )
        return tuple(out)

    def restricted(self, conditions: Sequence[Condition]) -> "RestrictedView":
        conds = self._validate_conditions(conditions)
        factor = self.root.restrict(conds)
        return RestrictedView(self, factor, list(conds.values()))

    # ---------------- 来源追溯 ---------------- #
    def lineage(self, tuple_value: Any):
        """返回完整输出元组的来源表达（LineageExpression）。

        不存在的输出元组得到根为 None 的空表达（exists() 为 False）；
        类型错误或值不属于输出列时抛 QueryError。惰性导入 lineage 模块，
        避免查询层与追溯层循环依赖。
        """
        from .lineage import lineage_of
        return lineage_of(self, tuple_value)

    def source_row(self, alias: str, row_seq: int):
        """按装载序号取实例来源行（供审计核对），序号越界返回 None。"""
        return self.store.fetch_source_row(alias, row_seq)

    def source_row_count(self, alias: str) -> int:
        """实例来源表的装载行数（来源行序号空间 0..n-1）。"""
        return self.store.source_row_count(alias)

    def _validate_conditions(
        self, conditions: Sequence[Condition]
    ) -> Dict[str, Condition]:
        if not conditions:
            return {}
        out: Dict[str, Condition] = {}
        for cond in conditions:
            if not isinstance(cond, Condition):
                raise QueryError("条件必须是 Condition 实例")
            dtype, nullable = self.store.attr_meta(cond.attr)
            if cond.attr in out:
                raise QueryError(f"列 {cond.attr!r} 上有重复条件")
            if cond.op == "eq":
                v = cond.value
                if dtype == "int64":
                    if isinstance(v, bool) or not isinstance(v, int):
                        raise QueryError(f"{cond.attr!r} 需要 int64 值")
                elif dtype == "text":
                    if not isinstance(v, str):
                        raise QueryError(f"{cond.attr!r} 需要 text 值")
                else:
                    if not isinstance(v, bool):
                        raise QueryError(f"{cond.attr!r} 需要 bool 值")
            elif cond.op == "is_null":
                if not nullable:
                    raise QueryError(f"列 {cond.attr!r} 不可为空")
            else:
                raise QueryError(f"不支持的操作 {cond.op!r}")
            out[cond.attr] = cond
        return out


class RestrictedView:
    """条件筛选后的结果：语义上重新计算连续秩。"""

    def __init__(self, session: QuerySession, factor: RNode,
                 conditions: List[Condition]):
        self.session = session
        self.factor = factor
        self.conditions = conditions
        self.output_attrs = session.output_attrs

    def count(self) -> int:
        return self.factor.weight

    def row_at(self, r: int) -> Tuple[Any, ...]:
        if not isinstance(r, int) or isinstance(r, bool) or r < 0:
            raise QueryError("r 必须是非负整数")
        if r >= self.factor.weight:
            raise QueryError(f"秩 {r} 越界（筛选后共 {self.factor.weight} 行）")
        out: List[Any] = []
        self.factor.row_at(r, out)
        if len(out) != len(self.output_attrs):
            raise ProtocolError("受限寻址的值个数与输出列不符")
        return tuple(out)

    def page(self, r: int, n: int) -> List[Tuple[Any, ...]]:
        if not isinstance(r, int) or isinstance(r, bool) or r < 0:
            raise QueryError("r 必须是非负整数")
        if not isinstance(n, int) or isinstance(n, bool) or n < 0:
            raise QueryError("n 必须是非负整数")
        total = self.count()
        if n == 0 or r >= total:
            return []
        end = min(r + n, total)
        return [self.row_at(i) for i in range(r, end)]
