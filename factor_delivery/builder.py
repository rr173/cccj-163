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


@dataclass
class BEmpty(BNode):
    def __init__(self, scope: Tuple[str, ...]):
        super().__init__(weight=0, scope=scope, kind="empty")


@dataclass
class BLeaf(BNode):
    def __init__(self, weight: int):
        super().__init__(weight=weight, scope=(), kind="leaf")


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
               scope: Optional[Tuple[str, ...]] = None) -> BNode:
    """构造袋并节点；空分支剔除，单分支直接返回。"""
    non_empty = [c for c in children if c.kind != "empty"]
    if not non_empty:
        return BEmpty(scope if scope is not None else children[0].scope)
    if len(non_empty) == 1:
        return non_empty[0]
    sc = scope if scope is not None else non_empty[0].scope
    weight = sum(c.weight for c in non_empty)
    node = BUnion.__new__(BUnion)
    BNode.__init__(node, weight=weight, scope=tuple(sc), kind="union")
    node.children = non_empty
    return node


# --------------------------------------------------------------------- #
def make_product(children: Sequence[BNode], output_pos: Dict[str, int]) -> BNode:
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
    for ch in flat:
        if not ch.scope:
            if ch.kind != "leaf":
                raise BuildError(f"空作用域因子只能是叶子，得到 {ch.kind}")
            empty_scope_weight *= ch.weight
        else:
            scoped.append(ch)
    if not scoped:
        # 整个乘积没有任何输出列：退化成一个承载合并重数的叶子
        return BLeaf(empty_scope_weight)
    if empty_scope_weight > 1:
        # 把纯重数缩放作用到第一个有作用域分支（等价于整体 multiplicity 乘积，
        # 不改变任何列结构或全序）
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


def union(a: Optional[BNode], b: BNode) -> BNode:
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
        return BLeaf(a.weight + b.weight)
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
                merged[k] = union(merged[k], child)
            else:
                merged[k] = child
                value_of[k] = value
        # 重新交给 BGroup 构造，按全序统一排序（NULL 最先）
        entries = [(value_of[k], c) for k, c in merged.items()]
        return BGroup(a.attr, entries)
    if a.kind == "product" and b.kind == "product":
        if len(a.children) != len(b.children) or a.scope != b.scope:
            raise BuildError("合并 product 的结构不一致")
        return make_product([union(x, y) for x, y in zip(a.children, b.children)],
                            _scope_positions(a.scope))
    raise BuildError(f"无法合并不同结构的因子: {a.kind} vs {b.kind}")


def union_all(nodes: Sequence[BNode]) -> BNode:
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
        return BLeaf(sum(n.weight for n in non_empty))
    if head.kind == "group":
        attr = head.attr
        if any(n.kind != "group" or n.attr != attr for n in non_empty):
            # 结构不同构：退化为逐对 union（仅出现在可空前缀等罕见路径）
            acc: Optional[BNode] = None
            for n in nodes:
                acc = n if acc is None else union(acc, n)
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
            (value_of[k], union_all(children))
            for k, children in buckets.items()
        ]
        return BGroup(attr, entries)
    if head.kind == "product":
        if any(n.kind != "product" for n in non_empty):
            acc = None
            for n in nodes:
                acc = n if acc is None else union(acc, n)
            assert acc is not None
            return acc
        nch = len(head.children)
        if any(len(n.children) != nch for n in non_empty):
            raise BuildError("袋并 product 的分支数不一致")
        return make_product(
            [union_all([n.children[i] for n in non_empty]) for i in range(nch)],
            _scope_positions(head.scope),
        )
    raise BuildError(f"未知节点类型 {head.kind}")


def _scope_positions(scope: Tuple[str, ...]) -> Dict[str, int]:
    return {a: i for i, a in enumerate(scope)}


def normalize(node: BNode, target_scope: Tuple[str, ...],
              output_pos: Dict[str, int]) -> BNode:
    """把因子的 group 嵌套/乘积分支按 target_scope（输出顺序）规范化。

    利用分组交换律在条件独立、作用域不重叠的前提下调整嵌套层；
    若目标顺序要求把被同一分组约束的属性与外部属性交错，则抛 BuildError。
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
            [normalize(ch, target_scope, output_pos) for ch in node.children],
            target_scope,
        )
    if node.kind == "product":
        # 各分支分别规范化，再按目标顺序排列；分支作用域必须各自构成
        # 目标顺序中的连续区间，否则无法在不展开的前提下保持全序
        rebuilt = [
            normalize(ch, tuple(sorted(ch.scope, key=lambda a: output_pos[a])),
                      output_pos)
            for ch in node.children
        ]
        rebuilt.sort(key=lambda c: min(output_pos[a] for a in c.scope))
        return make_product(rebuilt, output_pos)
    if node.kind == "group":
        return _normalize_group(node, target_scope, output_pos)
    raise BuildError(f"未知节点类型 {node.kind}")


def _normalize_group(node: BGroup, target_scope: Tuple[str, ...],
                     output_pos: Dict[str, int]) -> BNode:
    want_attr = target_scope[0]
    if node.attr == want_attr:
        # 最外层已正确：递归规范化每个子因子到剩余作用域
        rest = target_scope[1:]
        new_entries = [
            (value, normalize(child, rest, output_pos))
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
        inner = make_union([p[1] for p in pairs], rest_scope)
        inner = normalize(inner, rest_scope, output_pos)
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
    """把整棵因子的叶子重数放大 factor 倍（源 multiplicity 的乘积入口）。"""
    if factor == 1 or node.kind == "empty":
        return node
    if node.kind == "leaf":
        return BLeaf(node.weight * factor)
    if node.kind == "group":
        return BGroup(node.attr, [(v, scale(c, factor)) for v, c in node.entries])
    if node.kind == "product":
        return BProduct([scale(c, factor) for c in node.children])
    if node.kind == "union":
        return make_union([scale(c, factor) for c in node.children], node.scope)
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
        self.node_ids: Dict[int, int] = {}
        self.kind_counts = {"group": 0, "product": 0, "leaf": 0,
                           "empty": 0, "union": 0}
        self.group_entry_count = 0
        self.product_child_count = 0
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
        root = normalize(root, tuple(self.output), self.output_pos)
        root.scope = tuple(self.output)
        return root

    def _create_reduction_tables(self) -> None:
        for alias, tname in self.spec.instances.items():
            tbl = self.spec.tables[tname]
            ncols = len(tbl.columns)
            red = self.store.create_reduction_table(alias, ncols)
            src = self.store.src_physical(tname)
            cols = ", ".join(f"p{i}" for i in range(ncols))
            self.c.execute(
                f"INSERT INTO {red} ({cols}, mult) SELECT {cols}, mult FROM {src}"
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
                f"SELECT {select_cols}, mult FROM {red}"
            ).fetchall()
        else:
            key_pos = self._edge_positions(alias, edge)
            conds = " AND ".join(f"p{p} = ?" for p in key_pos)
            rows = self.c.execute(
                f"SELECT {select_cols}, mult FROM {red} WHERE {conds}",
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
            group_pos = [self.pos_of_attr[alias][a] for a in group_attrs]

            newly_bound = set(group_attrs)
            child_bound = tuple(set(bound) | newly_bound)

            # 一次性 Trie 装配。节点 = {"children": {值身份: 子}, "leaf": 载荷}。
            root_trie: Dict[str, Any] = {"children": {}, "leaf": None}
            for row in rows:
                mult = row[-1]
                values = row[:-1]
                cur = root_trie
                for phys in group_pos:
                    val = values[phys]
                    kids = cur["children"]
                    key = _value_identity(val)
                    nxt = kids.get(key)
                    if nxt is None:
                        nxt = {"children": {}, "leaf": None, "_v": val}
                        kids[key] = nxt
                    cur = nxt
                # 同一 trie 叶子的行在所有分组属性（含子键）上取值相同，
                # 因此对每个子实例拿到的是同一记忆化因子，只需累加 multiplicity。
                if cur["leaf"] is None:
                    cur["leaf"] = {"w": 0, "by_child": {}}
                payload = cur["leaf"]
                payload["w"] += mult
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
                factors: List[BNode] = [BLeaf(payload["w"])]
                factors.extend(payload["by_child"].values())
                return make_product(factors, self.output_pos)

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
                    group = union_all([group, leaf_factor(trie_node["leaf"])])
                return group

            if not group_attrs:
                # 无输出列也无子键：所有行折叠为单个重数叶子
                total_w = sum(r[-1] for r in rows)
                result: BNode = BLeaf(total_w)
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
            return make_union(projected, projected[0].scope)
        if node.kind == "product":
            return make_product(
                [self._project_outputs(c) for c in node.children],
                self.output_pos,
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
            )
        raise BuildError(f"未知节点类型 {node.kind}")

    # ---------------- 持久化 ---------------- #
    def persist(self, root: BNode) -> int:
        self.store.drop_factor_tables()
        root_id = self._write_node(root)
        self.store.write_build_stat("node_group", self.kind_counts["group"])
        self.store.write_build_stat("node_product", self.kind_counts["product"])
        self.store.write_build_stat("node_union", self.kind_counts["union"])
        self.store.write_build_stat("node_leaf", self.kind_counts["leaf"])
        self.store.write_build_stat("node_empty", self.kind_counts["empty"])
        self.store.write_build_stat("group_entries", self.group_entry_count)
        self.store.write_build_stat("product_children", self.product_child_count)
        self.store.write_build_stat("source_rows_total", self.source_rows)
        for alias, n in self.reduced_rows.items():
            self.store.write_build_stat(f"reduced_rows:{alias}", n)
        self.store.write_build_stat("max_physical_table_rows",
                                    self.max_physical_rows)
        self.store.write_build_stat("distinct_subtree_factors", len(self.memo) + 1)
        self.c.execute(
            "INSERT INTO fd_meta VALUES ('root_node_id', ?)", (str(root_id),)
        )
        return root_id

    def _write_node(self, node: BNode) -> int:
        key = id(node)
        existing = self.node_ids.get(key)
        if existing is not None:
            return existing
        nid = len(self.node_ids) + 1
        self.node_ids[key] = nid
        self.kind_counts[node.kind] += 1
        scope = ",".join(node.scope)
        self.c.execute(
            "INSERT INTO fd_node (node_id, kind, weight, scope) VALUES (?, ?, ?, ?)",
            (nid, node.kind, str(node.weight), scope),
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
