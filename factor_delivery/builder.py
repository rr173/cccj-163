"""因子结构构建器。

两个阶段，全程不展开完整连接：

1. Yannakakis 式半连接约减：在每个实例的 *独立* 物理表 red_<alias> 上，
   自叶向根、再自根向叶各做一遍 NOT EXISTS 半连接，消除无效分支。
   等值用 SQL ``=``，NULL = NULL 结果不是 TRUE，因此 null 键天然不匹配。
2. 递归构建因子 DAG：
   * GroupNode  —— 一个逻辑属性的排序值域，每个取值绑定一个子因子；
   * ProductNode —— 仅在父键已固定、分支条件独立时组合 *不重叠* 的属性作用域；
   * LeafNode   —— 一袋已固定的元组，weight 为合并后的 multiplicity；
   * EmptyNode  —— 空分支（总重数 0）。
   子树因子按 (实例, 父键值) 记忆化复用；同一来源表的不同角色走不同实例，
   身份不混淆。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .spec import Spec, Edge
from .store import DeliveryStore
from .provenance import PNode, PSrc, ProvFactory


class BuildError(RuntimeError):
    """构建失败（空结果以外的结构性问题，如输出顺序与连接树不兼容）。"""


# --------------------------------------------------------------------- #
# 构建期因子节点
# --------------------------------------------------------------------- #
@dataclass
class BNode:
    weight: int
    scope: Tuple[str, ...]
    kind: str = "?"
    # 来源闭包：该因子所有逻辑行的来源表达。empty 节点为 None；
    # 非空节点在最终闭包计算时由 ProvFactory 驻留赋值。
    prov: Optional[PNode] = None
    # 增量更新种子化节点：非 None 时持久化直接复用该旧 node_id，
    # 不重写节点及其整棵子树（干净分区的压缩节点保持原样）。
    persisted_id: Optional[int] = None


@dataclass
class BEmpty(BNode):
    def __init__(self, scope: Tuple[str, ...]):
        super().__init__(weight=0, scope=scope, kind="empty", prov=None)


@dataclass
class BLeaf(BNode):
    def __init__(self, weight: int, prov: Optional[PNode] = None):
        super().__init__(weight=weight, scope=(), kind="leaf", prov=prov)


# 排序键注册表：构建开始时安装 spec.value_key，使所有 BGroup 在构造/
# 合并后立即按全序有序（NULL 最先，数值/码点/false<true）。
_KEY_FN = None


def _sort_key(attr: str, value: Any):
    if _KEY_FN is None:
        raise BuildError("排序键未初始化")
    return _KEY_FN(attr, value)


@dataclass
class BGroup(BNode):
    attr: str = ""
    # 已按全序排好的 (值, 子因子)；None 表示 NULL 且排在最前
    entries: List[Tuple[Any, BNode]] = field(default_factory=list)

    def __init__(self, attr: str, entries: List[Tuple[Any, BNode]],
                 key_fn=None):
        if not entries:
            raise BuildError("group 节点至少需要一个取值")
        fn = key_fn or _sort_key
        entries = sorted(entries, key=lambda item: fn(attr, item[0]))
        scope = (attr,) + entries[0][1].scope
        weight = sum(child.weight for _, child in entries)
        super().__init__(weight=weight, scope=scope, kind="group")
        self.attr = attr
        self.entries = entries


@dataclass
class BProduct(BNode):
    children: List[BNode] = field(default_factory=list)

    def __init__(self, children: List[BNode]):
        if len(children) < 2:
            raise BuildError("product 节点至少需要两个分支")
        scope: List[str] = []
        weight = 1
        for ch in children:
            scope.extend(ch.scope)
            weight *= ch.weight
        super().__init__(weight=weight, scope=tuple(scope), kind="product")
        self.children = children


@dataclass
class BUnion(BNode):
    """袋并：同作用域、互不条件独立的多个分支，重数相加。

    用于投影掉连接键后保留各父键分支的正确语义——绝不把不同父键下的
    候选重新做成无条件笛卡尔积。
    """

    children: List[BNode] = field(default_factory=list)


def make_union(children: Sequence[BNode],
               scope: Optional[Tuple[str, ...]] = None,
               factory: Optional[ProvFactory] = None) -> BNode:
    """构造袋并节点；空分支剔除，单分支直接返回。"""
    non_empty = [c for c in children if c.kind != "empty"]
    if not non_empty:
        return BEmpty(scope if scope is not None else children[0].scope)
    if len(non_empty) == 1:
        return non_empty[0]
    sc = scope if scope is not None else non_empty[0].scope
    weight = sum(c.weight for c in non_empty)
    node = BUnion.__new__(BUnion)
    prov = None
    if factory is not None:
        provs = [c.prov for c in non_empty if c.prov is not None]
        if len(provs) == len(non_empty):
            prov = factory.alt(provs)
    BNode.__init__(node, weight=weight, scope=tuple(sc), kind="union", prov=prov)
    node.children = non_empty
    return node


# --------------------------------------------------------------------- #
def make_product(children: Sequence[BNode], output_pos: Dict[str, int],
                 factory: Optional[ProvFactory] = None) -> BNode:
    """组合条件独立的分支；空分支传播为空，并校验作用域不重叠且区间连续。"""
    flat: List[BNode] = []
    for ch in children:
        if ch.kind == "empty":
            # 任一分支为空，整个乘积为空；作用域取各分支声明的期望作用域
            scope = tuple(a for c in children for a in c.scope)
            return BEmpty(scope)
        if ch.kind == "product":
            flat.extend(ch.children)
        else:
            flat.append(ch)
    # 作用域为空的因子（只剩 multiplicity 的叶子）不参与列结构，
    # 但其重数必须并入：把所有空作用域叶子的重数累乘为 scale。
    # 注意 product 展平后可能再次出现空作用域叶子，因此这里统一过滤。
    scoped: List[BNode] = []
    empty_scope_weight = 1
    empty_scope_prov: List[PNode] = []
    for ch in flat:
        if not ch.scope:
            if ch.kind != "leaf":
                raise BuildError(f"空作用域因子只能是叶子，得到 {ch.kind}")
            empty_scope_weight *= ch.weight
            if ch.prov is not None:
                empty_scope_prov.append(ch.prov)
        else:
            scoped.append(ch)
    if not scoped:
        # 整个乘积没有任何输出列：退化成一个承载合并重数的叶子。
        # 来源表达 = 各实例来源的连接相乘（无输出列也要逐实例保留身份）。
        prov = None
        if factory is not None and empty_scope_prov:
            prov = factory.prod(empty_scope_prov)
        return BLeaf(empty_scope_weight, prov)
    if empty_scope_prov:
        # 空作用域兄弟（没有输出列的子树，如只含连接键的实例）在乘积中只
        # 出现一次：把它们的来源表达整体并入 *第一个* 有作用域分支。
        # 乘进多个分支会让这些来源被重复计数（笛卡尔积语义要求每个因子恰好
        # 一次）。attach_provenance 信任此处预置的闭包，被折叠实例的身份
        # 因而不会在重建时丢失。weight = 分支原重数 * factor，属于连接相乘，
        # 不逐份复制来源节点。
        extra = factory.prod(empty_scope_prov) if len(empty_scope_prov) > 1 \
            else empty_scope_prov[0]
        if extra.weight != empty_scope_weight:
            raise BuildError("空作用域来源重数与缩放因子不一致")
        if factory is not None:
            scoped[0] = scale_with_prov(
                scoped[0], empty_scope_weight, extra, factory
            )
        elif empty_scope_weight > 1:
            scoped[0] = scale(scoped[0], empty_scope_weight)
    if len(scoped) == 1:
        return scoped[0]
    flat = scoped

    # 作用域不重叠
    all_attrs = [a for c in flat for a in c.scope]
    if len(set(all_attrs)) != len(all_attrs):
        raise BuildError(f"product 分支的输出属性重叠: {all_attrs}")
    # 每个子作用域自身连续，且拼接后在输出顺序上是一个连续区间
    for c in flat:
        pos = sorted(output_pos[a] for a in c.scope)
        if pos != list(range(pos[0], pos[-1] + 1)):
            raise BuildError(f"分支作用域在输出顺序中不连续: {c.scope}")
    positions = [output_pos[a] for a in all_attrs]
    if positions != list(range(min(positions), max(positions) + 1)):
        raise BuildError(
            "输出列顺序与连接树不兼容：乘积分支的属性被其他列隔开，"
            "无法在不展开组合的前提下保持该全序"
        )
    # 子分支按其首个输出属性的位置排列，使 product 的 scope 与全序一致，
    # 秩寻址写值顺序即输出列顺序
    flat.sort(key=lambda c: min(output_pos[a] for a in c.scope))
    return BProduct(flat)


def union(a: Optional[BNode], b: BNode,
          factory: Optional[ProvFactory] = None) -> BNode:
    """同作用域的两个因子做袋合并（投影后相同输出元组的重数在此相加）。"""
    if a is None:
        return b
    if a.kind == "empty":
        if a.scope != b.scope:
            raise BuildError("合并因子的作用域不一致")
        return b
    if b.kind == "empty":
        if a.scope != b.scope:
            raise BuildError("合并因子的作用域不一致")
        return a
    if a.kind == "leaf" and b.kind == "leaf":
        prov = _alt_of(a, b, factory)
        return BLeaf(a.weight + b.weight, prov)
    if a.kind == "group" and b.kind == "group":
        if a.attr != b.attr or a.scope != b.scope:
            raise BuildError("合并 group 的属性/作用域不一致")
        # 键用 (类型名, 值)：Python 中 True == 1、False == 0 且哈希相同，
        # 裸 dict 会让 bool 与 int 互相覆盖 entry。
        merged: Dict[Tuple[str, Any], BNode] = {}
        value_of: Dict[Tuple[str, Any], Any] = {}
        for value, child in a.entries:
            k = _value_identity(value)
            merged[k] = child
            value_of[k] = value
        for value, child in b.entries:
            k = _value_identity(value)
            if k in merged:
                merged[k] = union(merged[k], child, factory)
            else:
                merged[k] = child
                value_of[k] = value
        # 重新交给 BGroup 构造，按全序统一排序（NULL 最先）
        entries = [(value_of[k], c) for k, c in merged.items()]
        return BGroup(a.attr, entries)
    if a.kind == "product" and b.kind == "product":
        if len(a.children) != len(b.children) or a.scope != b.scope:
            raise BuildError("合并 product 的结构不一致")
        return make_product(
            [union(x, y, factory) for x, y in zip(a.children, b.children)],
            _scope_positions(a.scope), factory,
        )
    raise BuildError(f"无法合并不同结构的因子: {a.kind} vs {b.kind}")


def _alt_of(a: BNode, b: BNode,
            factory: Optional[ProvFactory]) -> Optional[PNode]:
    """两个同构袋并因子的来源表达做 alt（相加）；缺工厂时留给最终闭包计算。"""
    if factory is None or a.prov is None or b.prov is None:
        return None
    return factory.alt([a.prov, b.prov])


def union_all(nodes: Sequence[BNode],
              factory: Optional[ProvFactory] = None) -> BNode:
    """同作用域的多个因子一次性袋并。

    分组折叠场景下各因子同构（同 attr/scope），用字典把同值 entry 收集到
    一起再递归合并，每个 group 只构造/排序一次，整体与 entry 总数成线性。
    """
    non_empty = [n for n in nodes if n.kind != "empty"]
    if not non_empty:
        return nodes[0]
    if len(non_empty) == 1:
        return non_empty[0]
    head = non_empty[0]
    if head.kind == "leaf":
        if any(n.kind != "leaf" for n in non_empty):
            raise BuildError("袋并中 leaf 与非 leaf 混用")
        prov = None
        if factory is not None and all(n.prov is not None for n in non_empty):
            prov = factory.alt([n.prov for n in non_empty])
        return BLeaf(sum(n.weight for n in non_empty), prov)
    if head.kind == "group":
        attr = head.attr
        if any(n.kind != "group" or n.attr != attr for n in non_empty):
            # 结构不同构：退化为逐对 union（仅出现在可空前缀等罕见路径）
            acc: Optional[BNode] = None
            for n in nodes:
                acc = n if acc is None else union(acc, n, factory)
            assert acc is not None
            return acc
        buckets: Dict[Tuple[str, Any], List[BNode]] = {}
        value_of: Dict[Tuple[str, Any], Any] = {}
        for gn in non_empty:
            for value, child in gn.entries:
                k = _value_identity(value)
                buckets.setdefault(k, []).append(child)
                value_of[k] = value
        entries = [
            (value_of[k], union_all(children, factory))
            for k, children in buckets.items()
        ]
        return BGroup(attr, entries)
    if head.kind == "product":
        if any(n.kind != "product" for n in non_empty):
            acc = None
            for n in nodes:
                acc = n if acc is None else union(acc, n, factory)
            assert acc is not None
            return acc
        nch = len(head.children)
        if any(len(n.children) != nch for n in non_empty):
            raise BuildError("袋并 product 的分支数不一致")
        return make_product(
            [union_all([n.children[i] for n in non_empty], factory)
             for i in range(nch)],
            _scope_positions(head.scope), factory,
        )
    raise BuildError(f"未知节点类型 {head.kind}")


def _scope_positions(scope: Tuple[str, ...]) -> Dict[str, int]:
    return {a: i for i, a in enumerate(scope)}


def normalize(node: BNode, target_scope: Tuple[str, ...],
              output_pos: Dict[str, int],
              factory: Optional[ProvFactory] = None) -> BNode:
    """把因子的 group 嵌套/乘积分支按 target_scope（输出顺序）规范化。

    利用分组交换律在条件独立、作用域不重叠的前提下调整嵌套层；
    若目标顺序要求把被同一分组约束的属性与外部属性交错，则抛 BuildError。
    规范化只重排结构，来源闭包随节点保留（叶子自带、复合节点最终统一计算）。
    """
    if node.kind == "empty":
        return BEmpty(target_scope)
    if node.kind == "leaf":
        if target_scope:
            raise BuildError("叶子无法承载仍需输出的属性")
        return node
    if set(node.scope) != set(target_scope):
        raise BuildError(
            f"规范化作用域不匹配: {node.scope} vs {target_scope}"
        )
    if node.kind == "union":
        # 袋并的每个分支同作用域，分别规范化后仍为袋并
        return make_union(
            [normalize(ch, target_scope, output_pos, factory)
             for ch in node.children],
            target_scope, factory,
        )
    if node.kind == "product":
        # 各分支分别规范化，再按目标顺序排列；分支作用域必须各自构成
        # 目标顺序中的连续区间，否则无法在不展开的前提下保持全序
        rebuilt = [
            normalize(ch, tuple(sorted(ch.scope, key=lambda a: output_pos[a])),
                      output_pos, factory)
            for ch in node.children
        ]
        rebuilt.sort(key=lambda c: min(output_pos[a] for a in c.scope))
        return make_product(rebuilt, output_pos, factory)
    if node.kind == "group":
        return _normalize_group(node, target_scope, output_pos, factory)
    raise BuildError(f"未知节点类型 {node.kind}")


def _normalize_group(node: BGroup, target_scope: Tuple[str, ...],
                     output_pos: Dict[str, int],
                     factory: Optional[ProvFactory]) -> BNode:
    want_attr = target_scope[0]
    if node.attr == want_attr:
        # 最外层已正确：递归规范化每个子因子到剩余作用域
        rest = target_scope[1:]
        new_entries = [
            (value, normalize(child, rest, output_pos, factory))
            for value, child in node.entries
        ]
        return BGroup(node.attr, new_entries)
    # 需要把 want_attr 从子树提升到最外层（group 交换）。
    # 孩子可能是 union（投影折叠产生的袋并）：必须对 union 的 *每个* 分支
    # 分别定位 want 分组并保留袋并，绝不能只取一个分支而丢掉其余重数。
    lifted: Dict[Any, List[Tuple[Any, BNode]]] = {}
    rest_scope = tuple(a for a in target_scope if a != want_attr)
    for outer_value, child in node.entries:
        for branch in _union_branches(child):
            sub = _locate_attr(branch, want_attr)
            if sub is None:
                raise BuildError(
                    f"输出顺序不兼容：属性 {want_attr!r} 与 {node.attr!r} "
                    f"被同一分组约束，无法交换"
                )
            for want_value, inner in sub.entries:
                # inner 仍可能含与 branch 同级的袋并，包上原外层 group
                wrapped = BGroup(node.attr, [(outer_value, inner)])
                lifted.setdefault(want_value, []).append(
                    (outer_value, wrapped))
    new_entries: List[Tuple[Any, BNode]] = []
    for want_value, pairs in lifted.items():
        # 同一 want 值下来自不同分支/外层值的因子是袋并关系
        inner = make_union([p[1] for p in pairs], rest_scope, factory)
        inner = normalize(inner, rest_scope, output_pos, factory)
        new_entries.append((want_value, inner))
    return BGroup(want_attr, new_entries)


def _union_branches(node: BNode) -> List[BNode]:
    """展平 union，返回袋并的所有分支；非 union 节点返回自身单元素列表。"""
    if node.kind == "union":
        out: List[BNode] = []
        for c in node.children:
            out.extend(_union_branches(c))
        return out
    return [node]


def _locate_attr(node: BNode, attr: str) -> Optional[BGroup]:
    """在因子中找到以 attr 分组的最浅 group；若 attr 被锁在不可交换位置则 None。"""
    if node.kind == "group":
        if node.attr == attr:
            return node
        # 仅当所有 entry 的孩子都能定位到同构 group 时才可继续
        found: Optional[BGroup] = None
        for _, child in node.entries:
            sub = _locate_attr(child, attr)
            if sub is None:
                return None
            if found is not None and sub.attr != found.attr:
                return None
            found = sub
        return found
    if node.kind == "product":
        # 属性若在某个独立分支内，可在该分支内提升
        for child in node.children:
            if attr in child.scope:
                return _locate_attr(child, attr)
        return None
    if node.kind == "union":
        # 袋并各分支同作用域；只有所有分支都能同构定位时才可交换
        found = None
        for child in node.children:
            sub = _locate_attr(child, attr)
            if sub is None:
                return None
            if found is not None and sub.attr != found.attr:
                return None
            found = sub
        return found
    return None


def _value_identity(value: Any) -> Tuple[str, Any]:
    """dict 键身份：区分 bool 与 int（True/False 与 1/0 哈希冲突）。

    None 单独成类；其余按 Python 类型名 + 值。
    """
    if value is None:
        return ("null", None)
    if isinstance(value, bool):
        return ("bool", value)
    return (type(value).__name__, value)


def scale(node: BNode, factor: int) -> BNode:
    """把整棵因子的叶子重数放大 factor 倍（源 multiplicity 的乘积入口）。

    复制结构、不改动入参节点（入参可能是被记忆化共享的子树）。
    来源闭包随结构复制保留，最终闭包计算时按新重数交叉核对。
    """
    if factor == 1 or node.kind == "empty":
        return node
    if node.kind == "leaf":
        return BLeaf(node.weight * factor, node.prov)
    if node.kind == "group":
        return BGroup(node.attr, [(v, scale(c, factor)) for v, c in node.entries])
    if node.kind == "product":
        return BProduct([scale(c, factor) for c in node.children])
    if node.kind == "union":
        return make_union([scale(c, factor) for c in node.children], node.scope)
    raise BuildError(f"未知节点类型 {node.kind}")


def scale_with_prov(node: BNode, factor: int, extra_prov: PNode,
                    factory: ProvFactory) -> BNode:
    """重数放大 factor 倍，并把空作用域兄弟的来源表达 extra_prov 并入每个叶子。

    extra_prov 的 weight 已按 factor 放大（见 make_product）。对叶子：
    原来源与兄弟来源在同一连接组合中共同贡献 -> prod；重数 = 原重数 * factor。
    """
    if node.kind == "leaf":
        prov = factory.prod([node.prov, extra_prov]) if node.prov is not None \
            else extra_prov
        return BLeaf(node.weight * factor, prov)
    if node.kind == "group":
        return BGroup(node.attr,
                      [(v, scale_with_prov(c, factor, extra_prov, factory))
                       for v, c in node.entries])
    if node.kind == "product":
        return BProduct(
            [scale_with_prov(c, factor, extra_prov, factory)
             for c in node.children]
        )
    if node.kind == "union":
        return make_union(
            [scale_with_prov(c, factor, extra_prov, factory)
             for c in node.children],
            node.scope, factory,
        )
    raise BuildError(f"未知节点类型 {node.kind}")


# --------------------------------------------------------------------- #
# 构建主流程
# --------------------------------------------------------------------- #
def build_factorized(store: DeliveryStore, spec: Spec) -> int:
    """在已装载规格的 store 上构建因子结构，返回根节点 id。"""
    global _KEY_FN
    spec.validate()
    _KEY_FN = spec.value_key
    builder = _Builder(store, spec)
    root = builder.run()
    builder.attach_provenance(root)
    root_id = builder.persist(root)
    store.write_build_stat("root_node_id", root_id)
    store.write_build_stat("logical_total_weight", root.weight)
    store.write_build_stat("sql_statements", store.conn.statements)
    store.write_build_stat("rows_inserted", store.conn.rows_inserted)
    store.conn.commit()
    return root_id


class _Builder:
    def __init__(self, store: DeliveryStore, spec: Spec):
        self.store = store
        self.spec = spec
        self.c = store.conn
        self.factory = ProvFactory()
        self.output = tuple(spec.output_attrs)
        self.output_pos = {a: i for i, a in enumerate(self.output)}
        # 实例物理信息
        self.cols: Dict[str, List[Tuple[str, int, str]]] = {}
        for alias in spec.instances:
            self.cols[alias] = [
                (col.attr, pos, col.dtype)
                for pos, col in enumerate(spec.instance_columns(alias))
            ]
        self.pos_of_attr: Dict[str, Dict[str, int]] = {
            alias: {attr: pos for attr, pos, _ in cols}
            for alias, cols in self.cols.items()
        }
        self._orient_tree()
        self.memo: Dict[Tuple[str, Tuple[Any, ...]], BNode] = {}
        # 每个 (实例, 父键) 记忆化子树因子的根（规范化前），持久化时登记到
        # fd_subtree_factor，供增量更新按分区种子化复用。
        self.subtree_roots: List[Tuple[str, Optional[Tuple[Any, ...]], BNode]] = []
        self.node_ids: Dict[int, int] = {}
        self.kind_counts = {"group": 0, "product": 0, "leaf": 0,
                           "empty": 0, "union": 0}
        self.group_entry_count = 0
        self.product_child_count = 0
        self.prov_counts = {"src": 0, "prod": 0, "alt": 0}
        self.prov_src_rows = 0
        self.prov_comp_rows = 0
        self.source_rows = 0
        self.reduced_rows: Dict[str, int] = {}
        self.max_physical_rows = 0

    # ---------------- 树定向 ---------------- #
    def _orient_tree(self) -> None:
        root = self.spec.root_alias()
        self.root_alias = root
        self.parent_edge: Dict[str, Optional[Edge]] = {root: None}
        self.parent_side: Dict[str, str] = {}
        self.children: Dict[str, List[str]] = {a: [] for a in self.spec.instances}
        self.child_edge: Dict[Tuple[str, str], Edge] = {}
        stack = [root]
        seen = {root}
        while stack:
            u = stack.pop()
            for v, edge in self.spec.adj[u]:
                if v in seen:
                    continue
                seen.add(v)
                self.parent_edge[v] = edge
                self.parent_side[v] = "L" if edge.alias_left == u else "R"
                self.children[u].append(v)
                self.child_edge[(u, v)] = edge
                stack.append(v)
        self.postorder: List[str] = []
        self._post(root)

    def _post(self, u: str) -> None:
        for v in self.children[u]:
            self._post(v)
        self.postorder.append(u)

    # ---------------- 阶段 1：约减 ---------------- #
    def run(self) -> BNode:
        self._create_reduction_tables()
        self._reduce()
        root = self.build_subtree(self.root_alias, None)
        if root.kind == "empty":
            return BEmpty(self.output)
        if set(root.scope) != set(self.output) or len(root.scope) != len(self.output):
            raise BuildError(
                f"根因子作用域 {root.scope} 与输出列 {self.output} 不一致"
            )
        # 按输出列顺序规范化 group 嵌套与乘积分支；不兼容顺序在此抛错
        root = normalize(root, tuple(self.output), self.output_pos, self.factory)
        root.scope = tuple(self.output)
        return root

    # ---------------- 来源闭包 ---------------- #
    def make_closure(self):
        """构造来源闭包计算器（规范化完成后对根及各子树因子分别调用）。"""
        def closure(node: BNode) -> Optional[PNode]:
            if node.kind == "empty":
                node.prov = None
                return None
            # 已预置来源闭包的节点直接信任：scale_with_prov 会把空作用域
            # 兄弟来源并入具体子树（记忆化子树被多个父键上下文复用时，各自
            # 得到自己的闭包），重算反而会丢掉这些被折叠的实例身份。
            if node.prov is not None:
                if node.prov.weight != node.weight:
                    raise BuildError(
                        f"预置来源闭包重数 {node.prov.weight} 与因子重数 "
                        f"{node.weight}（{node.kind}）不符"
                    )
                return node.prov
            if node.kind == "leaf":
                raise BuildError("非空叶子缺少来源表达")
            if node.kind == "group":
                # 空分区孩子没有来源闭包（增量重建后部分键分区可能变空），
                # 袋并只聚合非空孩子。
                child_provs = [p for _, c in node.entries
                               if (p := closure(c)) is not None]
                if not child_provs:
                    node.prov = None
                    return None
                prov = self.factory.alt(child_provs)  # type: ignore[arg-type]
            elif node.kind == "union":
                child_provs = [closure(c) for c in node.children]
                if any(p is None for p in child_provs):
                    raise BuildError("union 存在无来源表达的非空孩子")
                prov = self.factory.alt(child_provs)  # type: ignore[arg-type]
            elif node.kind == "product":
                child_provs = [closure(c) for c in node.children]
                if any(p is None for p in child_provs):
                    raise BuildError("product 存在无来源表达的非空孩子")
                prov = self.factory.prod(child_provs)  # type: ignore[arg-type]
            else:
                raise BuildError(f"未知节点类型 {node.kind}")
            if prov.weight != node.weight:
                raise BuildError(
                    f"来源闭包重数 {prov.weight} 与因子重数 {node.weight} "
                    f"（{node.kind}）不符"
                )
            node.prov = prov
            return prov
        return closure

    def attach_provenance(self, root: BNode) -> None:
        """规范化完成后，为整棵因子 DAG 计算并驻留来源闭包。

        叶子的来源在 Trie 装配时已带；复合节点的闭包按结构合成：
        group/union -> 各孩子 alt（袋并相加），product -> 各孩子 prod（相乘）。
        全程只遍历因子节点，不枚举任何来源组合。合成后逐节点核对
        来源 weight 与因子 weight 必须相等。
        """
        closure = self.make_closure()
        if root.kind != "empty":
            closure(root)
        # 每个记忆化子树因子也独立闭包：它们需要可独立持久化，供增量
        # 更新按分区种子化复用（已带闭包的直接跳过）。
        for _alias, _key, sub in self.subtree_roots:
            if sub.kind != "empty" and sub.prov is None:
                closure(sub)

    def _create_reduction_tables(self) -> None:
        for alias, tname in self.spec.instances.items():
            tbl = self.spec.tables[tname]
            ncols = len(tbl.columns)
            red = self.store.create_reduction_table(alias, ncols)
            src = self.store.src_physical(tname)
            cols = ", ".join(f"p{i}" for i in range(ncols))
            # 连同 row_seq 一起复制：约减只删行，装载序号保持不变。
            # 增量版本的源表带 live 标记，只复制存活记录（撤销行不参与）。
            self.c.execute(
                f"INSERT INTO {red} (row_seq, {cols}, mult) "
                f"SELECT row_seq, {cols}, mult FROM {src} WHERE live=1"
            )
            cur = self.c.execute(f"SELECT COUNT(*) FROM {src}")
            n = cur.fetchone()[0]
            self.source_rows += n
            self.max_physical_rows = max(self.max_physical_rows, n)
        # 每条树边的两端各建一个多列索引
        made: set = set()
        self._index_seq = 0
        for u in self.spec.instances:
            edge = self.parent_edge[u]
            if edge is None:
                continue
            self._index_edge(u, edge, made)
            parent = self._other(u, edge)
            self._index_edge(parent, edge, made)

    def _index_edge(self, alias: str, edge: Edge, made: set) -> None:
        red = self.store.red_physical(alias)
        positions = tuple(self._edge_positions(alias, edge))
        key = (alias, positions)
        if key in made:
            return
        made.add(key)
        idx_cols = ", ".join(f"p{p}" for p in positions)
        self._index_seq += 1
        idx_name = f"idx_{red}_{self._index_seq}"
        self.c.execute(f"CREATE INDEX {idx_name} ON {red} ({idx_cols})")

    def _edge_positions(self, alias: str, edge: Edge) -> List[int]:
        if alias == edge.alias_left:
            cols = edge.cols_left
        elif alias == edge.alias_right:
            cols = edge.cols_right
        else:
            raise BuildError("边不属于该实例")
        tname = self.spec.instances[alias]
        tbl = self.spec.tables[tname]
        return [self.pos_of_attr[alias][tbl.col(name).attr] for name in cols]

    def _other(self, alias: str, edge: Edge) -> str:
        a, b = edge.endpoints()
        return b if alias == a else a

    def _reduce(self) -> None:
        # 自叶向根：父表半连接每个已约减的子表
        for u in self.postorder:
            for v in self.children[u]:
                self._semijoin_filter(u, v, self.child_edge[(u, v)])
        # 自根向叶：子表半连接已约减的父表
        for u in reversed(self.postorder):
            for v in self.children[u]:
                self._semijoin_filter(v, u, self.child_edge[(u, v)])
        for alias in self.spec.instances:
            red = self.store.red_physical(alias)
            n = self.c.execute(f"SELECT COUNT(*) FROM {red}").fetchone()[0]
            self.reduced_rows[alias] = n
            self.max_physical_rows = max(self.max_physical_rows, n)

    def _semijoin_filter(self, target: str, other: str, edge: Edge) -> None:
        """从 red_target 中删除在 red_other 中没有严格等值匹配的行（含 null 键）。"""
        rt = self.store.red_physical(target)
        ro = self.store.red_physical(other)
        tp = self._edge_positions(target, edge)
        op = self._edge_positions(other, edge)
        conds = " AND ".join(f"o.p{o} = t.p{p}" for p, o in zip(tp, op))
        # NULL = NULL 不为 TRUE；NOT EXISTS 同时淘汰 null 键行
        self.c.execute(
            f"DELETE FROM {rt} AS t WHERE NOT EXISTS "
            f"(SELECT 1 FROM {ro} AS o WHERE {conds})"
        )

    # ---------------- 阶段 2：递归因子 ---------------- #
    def _expected_scope(self, alias: str,
                        bound: Tuple[str, ...] = ()) -> Tuple[str, ...]:
        """子树（含本实例）中 *尚未绑定* 的输出属性，按输出顺序排列。

        bound 是祖先（含连接边）已经固定的逻辑属性；它们不能再出现在
        子因子作用域里，否则乘积分支的属性就会重叠。
        """
        bound_set = set(bound)
        attrs: List[str] = []
        stack = [alias]
        while stack:
            u = stack.pop()
            for attr, _, _ in self.cols[u]:
                if attr in self.output_pos and attr not in bound_set \
                        and attr not in attrs:
                    attrs.append(attr)
            stack.extend(self.children[u])
        return tuple(sorted(attrs, key=lambda a: self.output_pos[a]))

    def build_subtree(self, alias: str,
                      parent_key: Optional[Tuple[Any, ...]],
                      bound: Tuple[str, ...] = ()) -> BNode:
        if parent_key is not None:
            memo_key = (alias, parent_key)
            cached = self.memo.get(memo_key)
            if cached is not None:
                return cached
        expected_scope = self._expected_scope(alias, bound)
        red = self.store.red_physical(alias)
        ncols = len(self.cols[alias])
        select_cols = ", ".join(f"p{i}" for i in range(ncols))

        edge = self.parent_edge[alias]
        if edge is None:
            rows = self.c.execute(
                f"SELECT row_seq, {select_cols}, mult FROM {red}"
            ).fetchall()
        else:
            key_pos = self._edge_positions(alias, edge)
            conds = " AND ".join(f"p{p} = ?" for p in key_pos)
            rows = self.c.execute(
                f"SELECT row_seq, {select_cols}, mult FROM {red} WHERE {conds}",
                parent_key,
            ).fetchall()

        if not rows:
            result: BNode = BEmpty(expected_scope)
        else:
            visible_attrs = [a for a, _, _ in self.cols[alias]
                             if a in self.output_pos and a not in bound]
            visible_attrs.sort(key=lambda a: self.output_pos[a])
            # 分组维度还必须包含各子边的连接键：它们决定与子实例的条件独立
            # 边界。即使键不输出，也要先按它分组，投影时再折叠，绝不能让
            # 不同父键下的候选被直接做成无条件笛卡尔积。
            child_key_attrs: List[str] = []
            for child_alias in self.children[alias]:
                cedge = self.child_edge[(alias, child_alias)]
                for a in self.spec.edge_attrs(cedge):
                    if a not in bound and a not in visible_attrs \
                            and a not in child_key_attrs:
                        child_key_attrs.append(a)
            group_attrs = visible_attrs + child_key_attrs
            # SELECT row_seq, p0..p{k-1}, mult：物理列值在 row 中的下标 +1
            group_pos = [self.pos_of_attr[alias][a] + 1 for a in group_attrs]

            newly_bound = set(group_attrs)
            child_bound = tuple(set(bound) | newly_bound)

            # 一次性 Trie 装配。节点 = {"children": {值身份: 子}, "leaf": 载荷}。
            root_trie: Dict[str, Any] = {"children": {}, "leaf": None}
            for row in rows:
                row_seq = row[0]
                mult = row[-1]
                values = row[1:-1]
                cur = root_trie
                for phys in group_pos:
                    val = values[phys - 1]
                    kids = cur["children"]
                    key = _value_identity(val)
                    nxt = kids.get(key)
                    if nxt is None:
                        nxt = {"children": {}, "leaf": None, "_v": val}
                        kids[key] = nxt
                    cur = nxt
                # 同一 trie 叶子的行在所有分组属性（含子键）上取值相同，
                # 因此对每个子实例拿到的是同一记忆化因子，只需累加 multiplicity。
                # 来源行按装载序号登记；值相同但序号不同的行分别保留。
                if cur["leaf"] is None:
                    cur["leaf"] = {"w": 0, "seqs": [], "by_child": {}}
                payload = cur["leaf"]
                payload["w"] += mult
                payload["seqs"].append((row_seq, mult))
                for child_alias in self.children[alias]:
                    cedge = self.child_edge[(alias, child_alias)]
                    ckey_pos = self._edge_positions(alias, cedge)
                    key_values = tuple(values[p] for p in ckey_pos)
                    child_scope = self._expected_scope(child_alias, child_bound)
                    if any(v is None for v in key_values):
                        cf: BNode = BEmpty(child_scope)
                    else:
                        cf = self.build_subtree(child_alias, key_values,
                                                child_bound)
                    # 同键 -> 同一记忆化对象；直接登记即可，身份不混
                    payload["by_child"][child_alias] = cf

            def leaf_factor(payload: Dict[str, Any]) -> BNode:
                # 本实例在该分组取值上的来源行集合（mult 保留为数据，不展开）
                own = BLeaf(
                    payload["w"],
                    self.factory.src(alias, payload["seqs"]),
                )
                factors: List[BNode] = [own]
                factors.extend(payload["by_child"].values())
                return make_product(factors, self.output_pos, self.factory)

            def assemble(trie_node: Dict[str, Any], depth: int) -> BNode:
                kids = trie_node["children"]
                if not kids:
                    return leaf_factor(trie_node["leaf"])
                attr = group_attrs[depth]
                entries = [
                    (child_t["_v"], assemble(child_t, depth + 1))
                    for child_t in kids.values()
                ]
                group: BNode = BGroup(attr, entries)
                if trie_node["leaf"] is not None:  # 前缀终止行（可空列）
                    group = union_all([group, leaf_factor(trie_node["leaf"])],
                                      self.factory)
                return group

            if not group_attrs:
                # 无输出列也无子键：所有行折叠为单个重数叶子，来源仍是本实例
                total_w = sum(r[-1] for r in rows)
                all_seqs = [(r[0], r[-1]) for r in rows]
                result: BNode = BLeaf(
                    total_w, self.factory.src(alias, all_seqs)
                )
            else:
                assembled = assemble(root_trie, 0)
                # 投影：折叠所有非输出的分组属性（内部连接键），做袋并
                result = self._project_outputs(assembled)

            if result.kind != "empty" and \
                    self._scope_sorted(result.scope) != expected_scope:
                raise BuildError(
                    f"实例 {alias!r} 子树作用域 {result.scope} 与期望 "
                    f"{expected_scope} 不符"
                )
            if result.kind == "empty":
                result = BEmpty(expected_scope)

        if parent_key is not None:
            self.memo[(alias, parent_key)] = result
            # 只登记按父键分区的记忆化子树因子；根实例（parent_key=None）
            # 是整棵树的未规范化根，不能进分区索引——它与最终规范化真根
            # 重复，登记进去会让 prune 保留一棵结构不一致的旧根。
            self.subtree_roots.append((alias, parent_key, result))
        return result

    def _scope_sorted(self, scope: Tuple[str, ...]) -> Tuple[str, ...]:
        """输出属性按定义顺序在前，非输出（内部键）稳定附后。"""
        out = [a for a in scope if a in self.output_pos]
        internal = [a for a in scope if a not in self.output_pos]
        out.sort(key=lambda a: self.output_pos[a])
        return tuple(out + internal)

    def _project_outputs(self, node: BNode) -> BNode:
        """自底向上折叠非输出属性的分组。

        非输出连接键的各取值分支是 *袋并*（重数相加），用 BUnion 保留，
        绝不重新分配成笛卡尔积——(a×b)∪(c×d) ≠ (a∪c)×(b∪d)。
        """
        if node.kind in ("leaf", "empty"):
            return node
        if node.kind == "union":
            projected = [self._project_outputs(c) for c in node.children]
            return make_union(projected, projected[0].scope, self.factory)
        if node.kind == "product":
            return make_product(
                [self._project_outputs(c) for c in node.children],
                self.output_pos, self.factory,
            )
        if node.kind == "group":
            projected_children = [
                (v, self._project_outputs(c)) for v, c in node.entries
            ]
            if node.attr in self.output_pos:
                return BGroup(node.attr, projected_children)
            # 内部连接键不输出：同作用域的各分支袋并（保留每个父键分支）
            return make_union(
                [c for _, c in projected_children],
                projected_children[0][1].scope,
                self.factory,
            )
        raise BuildError(f"未知节点类型 {node.kind}")

    # ---------------- 持久化 ---------------- #
    def persist(self, root: BNode, incremental: bool = False) -> int:
        if not incremental:
            self.store.drop_factor_tables()
        self._prov_ids: Dict[int, int] = {}
        # 增量更新可传入高水位起始 id；全量构建从 1 开始
        self._next_node_id = getattr(self, "node_id_start", 1)
        self._next_prov_id = getattr(self, "prov_id_start", 1)
        root_id = self._write_node(root)
        # 记忆化子树因子全部持久化并登记分区索引（增量更新的种子接缝）。
        # 增量模式下索引先清空再重写：种子分区写回旧 id，脏分区写新 id。
        if incremental:
            self.store.clear_subtree_factors()
        for alias, pkey, sub in self.subtree_roots:
            nid = self._write_node(sub)
            self.store.write_subtree_factor(
                alias, pkey if pkey is not None else (), nid
            )
        if not incremental:
            self.store.write_build_stat("node_group", self.kind_counts["group"])
            self.store.write_build_stat("node_product", self.kind_counts["product"])
            self.store.write_build_stat("node_union", self.kind_counts["union"])
            self.store.write_build_stat("node_leaf", self.kind_counts["leaf"])
            self.store.write_build_stat("node_empty", self.kind_counts["empty"])
            self.store.write_build_stat("group_entries", self.group_entry_count)
            self.store.write_build_stat("product_children",
                                        self.product_child_count)
            self.store.write_build_stat("prov_src", self.prov_counts["src"])
            self.store.write_build_stat("prov_prod", self.prov_counts["prod"])
            self.store.write_build_stat("prov_alt", self.prov_counts["alt"])
            self.store.write_build_stat("prov_src_rows", self.prov_src_rows)
            self.store.write_build_stat("prov_comp_rows", self.prov_comp_rows)
            self.store.write_build_stat("source_rows_total", self.source_rows)
            for alias, n in self.reduced_rows.items():
                self.store.write_build_stat(f"reduced_rows:{alias}", n)
            self.store.write_build_stat("max_physical_table_rows",
                                        self.max_physical_rows)
            self.store.write_build_stat("distinct_subtree_factors",
                                        len(self.memo) + 1)
            self.c.execute(
                "INSERT INTO fd_meta VALUES ('root_node_id', ?)", (str(root_id),)
            )
        return root_id

    def _write_node(self, node: BNode) -> int:
        # 种子化节点：复用旧 node_id，整棵子树（含来源闭包）原样保留，
        # 不写任何行——干净分区的压缩节点与来源身份因此逐字节延续。
        if node.persisted_id is not None:
            self.node_ids[id(node)] = node.persisted_id
            return node.persisted_id
        key = id(node)
        existing = self.node_ids.get(key)
        if existing is not None:
            return existing
        nid = self._next_node_id
        self._next_node_id += 1
        self.node_ids[key] = nid
        self.kind_counts[node.kind] += 1
        scope = ",".join(node.scope)
        prov_id = None
        if node.prov is not None:
            prov_id = self._write_prov(node.prov)
        self.c.execute(
            "INSERT INTO fd_node (node_id, kind, weight, scope, prov_id) "
            "VALUES (?, ?, ?, ?, ?)",
            (nid, node.kind, str(node.weight), scope, prov_id),
        )
        if node.kind == "group":
            self._write_group(nid, node)
        elif node.kind in ("product", "union"):
            for pos, child in enumerate(node.children):
                cid = self._write_node(child)
                self.c.execute(
                    "INSERT INTO fd_product_child VALUES (?, ?, ?)",
                    (nid, pos, cid),
                )
                self.product_child_count += 1
        return nid

    def _write_prov(self, prov: PNode) -> int:
        """驻留后的来源节点按对象身份分配稳定 id，共享结构只写一份。"""
        # 惰性引用：直接复用旧版本的 prov_id，不复制其内容
        if prov.kind == "lazy_ref":
            assert prov.prov_id is not None
            self._prov_ids[id(prov)] = prov.prov_id
            return prov.prov_id
        existing = self._prov_ids.get(id(prov))
        if existing is not None:
            return existing
        pid = self._next_prov_id
        self._next_prov_id += 1
        self._prov_ids[id(prov)] = pid
        self.prov_counts[prov.kind] += 1
        self.c.execute(
            "INSERT INTO fd_prov_node (prov_id, kind, weight) VALUES (?, ?, ?)",
            (pid, prov.kind, str(prov.weight)),
        )
        if prov.kind == "src":
            assert isinstance(prov, PSrc)
            for pos, (row_seq, mult) in enumerate(prov.rows):
                self.c.execute(
                    "INSERT INTO fd_prov_src "
                    "(prov_id, alias, row_seq, mult, position) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (pid, prov.alias, row_seq, mult, pos),
                )
                self.prov_src_rows += 1
        else:
            for pos, child in enumerate(prov.children):
                cid = self._write_prov(child)
                self.c.execute(
                    "INSERT INTO fd_prov_comp "
                    "(prov_id, position, child_prov) VALUES (?, ?, ?)",
                    (pid, pos, cid),
                )
                self.prov_comp_rows += 1
        return pid

    def _write_group(self, nid: int, node: BGroup) -> None:
        # 统一按规格定义的全序排序：NULL 最先，然后数值/码点/false<true。
        # 不能用 SQL ORDER BY，因为 bool/int 列里的 NULL 会干扰排序语义。
        ordered = sorted(
            node.entries,
            key=lambda item: self.spec.value_key(node.attr, item[0]),
        )
        dtype = self.spec.attr_type[node.attr]
        for ord_, (value, child) in enumerate(ordered):
            cid = self._write_node(child)
            is_null = value is None
            # 先按 dtype 选列；bool 是 int 子类，不能用 isinstance 猜测
            if is_null:
                v_int = v_text = v_bool = None
            elif dtype == "bool":
                v_int, v_text, v_bool = None, None, int(value)
            elif dtype == "int64":
                v_int, v_text, v_bool = value, None, None
            else:
                v_int, v_text, v_bool = None, value, None
            self.c.execute(
                "INSERT INTO fd_group_entry "
                "(node_id, ord, attr, is_null, v_int, v_text, v_bool, child_id) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (nid, ord_, node.attr, 1 if is_null else 0,
                 v_int, v_text, v_bool, cid),
            )
            self.group_entry_count += 1
