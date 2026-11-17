"""输入规格：有限关系表、列类型/可空标记、连接树、输出列。

逻辑属性（logical attribute）是列的 *身份*：不同表实例上的列只要 attr 相同，
就表示同一个逻辑属性。连接边按次序配对的列必须属于同一逻辑属性且类型相同。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

INT64_MIN = -(2**63)
INT64_MAX = 2**63 - 1
VALID_TYPES = ("int64", "text", "bool")


class SpecError(ValueError):
    """规格本身不合法（类型、树、运行交集等校验失败）。"""


@dataclass(frozen=True)
class Column:
    name: str
    dtype: str          # "int64" | "text" | "bool"
    nullable: bool
    attr: str           # 逻辑属性身份

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise SpecError("列名必须是非空字符串")
        if self.dtype not in VALID_TYPES:
            raise SpecError(f"列 {self.name!r} 类型非法: {self.dtype!r}")
        if not isinstance(self.attr, str) or not self.attr:
            raise SpecError(f"列 {self.name!r} 的逻辑属性名非法")


@dataclass
class Table:
    columns: List[Column]
    rows: List[Tuple[Any, ...]]
    multiplicities: Optional[List[int]] = None

    def __post_init__(self) -> None:
        names = [c.name for c in self.columns]
        if len(set(names)) != len(names):
            raise SpecError("表内列名重复")
        if self.multiplicities is None:
            self.multiplicities = [1] * len(self.rows)
        if len(self.multiplicities) != len(self.rows):
            raise SpecError("multiplicity 行数与数据行数不一致")
        for i, mult in enumerate(self.multiplicities):
            if not isinstance(mult, int) or isinstance(mult, bool) or mult <= 0:
                raise SpecError(f"第 {i} 行 multiplicity 必须是正整数")
        col_by_name = {c.name: c for c in self.columns}
        for ridx, row in enumerate(self.rows):
            if len(row) != len(self.columns):
                raise SpecError(f"第 {ridx} 行的列数与表定义不符")
            for c, v in zip(self.columns, row):
                _check_value(c, v, ridx)
        self._col_by_name = col_by_name

    def col(self, name: str) -> Column:
        return self._col_by_name[name]


def _check_value(c: Column, v: Any, ridx: int) -> None:
    if v is None:
        if not c.nullable:
            raise SpecError(f"列 {c.name!r} 不可为空，但第 {ridx} 行是 NULL")
        return
    if c.dtype == "int64":
        if isinstance(v, bool) or not isinstance(v, int):
            raise SpecError(f"列 {c.name!r} 需要 int64，第 {ridx} 行得到 {type(v).__name__}")
        if not INT64_MIN <= v <= INT64_MAX:
            raise SpecError(f"列 {c.name!r} 第 {ridx} 行超出 int64 范围")
    elif c.dtype == "text":
        if not isinstance(v, str):
            raise SpecError(f"列 {c.name!r} 需要 text，第 {ridx} 行得到 {type(v).__name__}")
    else:  # bool
        if not isinstance(v, bool):
            raise SpecError(f"列 {c.name!r} 需要 bool，第 {ridx} 行得到 {type(v).__name__}")


@dataclass(frozen=True)
class Edge:
    """无向连接边：left_cols[i] 与 right_cols[i] 等值配对（按次序）。"""

    alias_left: str
    cols_left: Tuple[str, ...]
    alias_right: str
    cols_right: Tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.cols_left or len(self.cols_left) != len(self.cols_right):
            raise SpecError("连接边两侧键列数量必须相等且非空")
        if len(set(self.cols_left)) != len(self.cols_left) or \
           len(set(self.cols_right)) != len(self.cols_right):
            raise SpecError("同一条边上的键列不能重复")

    def endpoints(self) -> Tuple[str, str]:
        return self.alias_left, self.alias_right


@dataclass
class Spec:
    tables: Dict[str, Table]
    instances: Dict[str, str]          # 角色别名 -> 来源表名（同一表可被多个角色复用）
    edges: List[Edge]
    output_attrs: List[str]
    name: str = "factor_delivery"

    # 校验后填充的派生结构
    adj: Dict[str, List[Tuple[str, Edge]]] = field(default_factory=dict)
    attr_type: Dict[str, str] = field(default_factory=dict)
    attr_nullable: Dict[str, bool] = field(default_factory=dict)
    attr_occurrences: Dict[str, List[str]] = field(default_factory=dict)

    def validate(self) -> None:
        self._check_tables_and_instances()
        self._check_edges()
        self._check_tree()
        self._check_attr_consistency()
        self._check_running_intersection()
        self._check_output()
        self._build_derived()

    # ------------------------------------------------------------------ #
    def _check_tables_and_instances(self) -> None:
        if not self.tables:
            raise SpecError("至少需要一张表")
        for tname, tbl in self.tables.items():
            if not isinstance(tname, str) or not tname:
                raise SpecError("表名必须是非空字符串")
            if not isinstance(tbl, Table):
                raise SpecError(f"表 {tname!r} 必须是 Table 实例")
        if not self.instances:
            raise SpecError("至少需要一个表实例")
        if len(set(self.instances)) != len(self.instances) or \
           any(a in self.tables for a in []):  # 别名与表名允许重名，身份以别名为准
            pass
        for alias, tname in self.instances.items():
            if not isinstance(alias, str) or not alias:
                raise SpecError("实例别名必须是非空字符串")
            if tname not in self.tables:
                raise SpecError(f"实例 {alias!r} 引用了不存在的表 {tname!r}")

    def _check_edges(self) -> None:
        for e in self.edges:
            a, b = e.endpoints()
            if a == b:
                raise SpecError(f"边 {a}--{b} 不能连接实例自身（自连接请用不同角色别名）")
            if a not in self.instances or b not in self.instances:
                raise SpecError(f"边引用了不存在的实例: {a}--{b}")
            ta = self.tables[self.instances[a]]
            tb = self.tables[self.instances[b]]
            for cl, cr in zip(e.cols_left, e.cols_right):
                ca, cb = ta.col(cl), tb.col(cr)
                if ca.attr != cb.attr:
                    raise SpecError(
                        f"边 {a}.{cl} -- {b}.{cr} 的逻辑属性不一致: "
                        f"{ca.attr!r} != {cb.attr!r}"
                    )
                if ca.dtype != cb.dtype:
                    raise SpecError(
                        f"边 {a}.{cl} -- {b}.{cr} 类型不一致: "
                        f"{ca.dtype} != {cb.dtype}"
                    )

    def _check_tree(self) -> None:
        self.adj = {a: [] for a in self.instances}
        for e in self.edges:
            a, b = e.endpoints()
            self.adj[a].append((b, e))
            self.adj[b].append((a, e))
        n = len(self.instances)
        if len(self.edges) != n - 1:
            raise SpecError(
                f"连接图必须是树：{n} 个节点应有 {n - 1} 条边，实际 {len(self.edges)} 条"
            )
        # 连通性（边数正确时，连通即树）
        seen = set()
        stack = [next(iter(self.instances))]
        while stack:
            u = stack.pop()
            if u in seen:
                continue
            seen.add(u)
            stack.extend(v for v, _ in self.adj[u] if v not in seen)
        if seen != set(self.instances):
            raise SpecError("连接图不连通，存在与树断开的实例")

    def _check_attr_consistency(self) -> None:
        self.attr_type = {}
        self.attr_nullable = {}
        self.attr_occurrences = {}
        for alias, tname in self.instances.items():
            for col in self.tables[tname].columns:
                if col.attr in self.attr_type:
                    if self.attr_type[col.attr] != col.dtype:
                        raise SpecError(
                            f"逻辑属性 {col.attr!r} 在不同实例上类型不一致"
                        )
                    self.attr_nullable[col.attr] = \
                        self.attr_nullable[col.attr] or col.nullable
                else:
                    self.attr_type[col.attr] = col.dtype
                    self.attr_nullable[col.attr] = col.nullable
                self.attr_occurrences.setdefault(col.attr, []).append(alias)
        # 边上共享的属性必须真的在边上配对：若边两端都含某 attr，
        # 该 attr 必须出现在这条边的键中（保证共享属性一次一致绑定）。
        for e in self.edges:
            a, b = e.endpoints()
            ta = self.tables[self.instances[a]]
            tb = self.tables[self.instances[b]]
            edge_attrs = {ta.col(c).attr for c in e.cols_left}
            attrs_a = {c.attr for c in ta.columns}
            attrs_b = {c.attr for c in tb.columns}
            for shared in attrs_a & attrs_b:
                if shared not in edge_attrs:
                    raise SpecError(
                        f"实例 {a!r} 与 {b!r} 都含逻辑属性 {shared!r}，"
                        f"但连接边未按该属性配对，无法保证一致绑定"
                    )

    def _check_running_intersection(self) -> None:
        """运行交集：每个逻辑属性的出现节点集合在树上连通。"""
        for attr, occ in self.attr_occurrences.items():
            if len(occ) <= 1:
                continue
            target = set(occ)
            # 从任一出现节点 BFS，只能经过同样含该属性的节点，看能否覆盖全部出现点
            start = occ[0]
            reached = {start}
            stack = [start]
            while stack:
                u = stack.pop()
                for v, _ in self.adj[u]:
                    if v in reached:
                        continue
                    u_attrs = {c.attr for c in
                               self.tables[self.instances[u]].columns}
                    v_attrs = {c.attr for c in
                               self.tables[self.instances[v]].columns}
                    if attr in u_attrs and attr in v_attrs:
                        reached.add(v)
                        stack.append(v)
            if not target.issubset(reached):
                raise SpecError(
                    f"逻辑属性 {attr!r} 违反运行交集：出现节点 {sorted(target)} "
                    f"在树上不经含该属性的节点就无法连通"
                )

    def _check_output(self) -> None:
        if len(set(self.output_attrs)) != len(self.output_attrs):
            raise SpecError("输出列存在重复的逻辑属性")
        for attr in self.output_attrs:
            if attr not in self.attr_type:
                raise SpecError(f"输出列 {attr!r} 未出现在任何表实例中")

    def _build_derived(self) -> None:
        # 再次确保派生结构就绪（validate 各步骤已填充，这里仅兜底）
        if not self.adj:
            self._check_tree()

    # ------------------------------------------------------------------ #
    def root_alias(self) -> str:
        """任选一个稳定的根（别名字典序最小），树任意根均可。"""
        return next(iter(sorted(self.instances)))

    def instance_columns(self, alias: str) -> List[Column]:
        return list(self.tables[self.instances[alias]].columns)

    def edge_attrs(self, e: Edge) -> List[str]:
        ta = self.tables[self.instances[e.alias_left]]
        return [ta.col(c).attr for c in e.cols_left]

    def value_key(self, attr: str, v: Any) -> Tuple[Any, ...]:
        """全序键：NULL 先于非 NULL；int 数值序；text 码点序；bool false<true。

        三维结构 (null_flag, type 内序数, 载荷)。注意必须按属性 *声明类型*
        分派：bool 经 SQLite 整数列往返后运行时类型会变成 int，不能用
        isinstance(v, bool) 判断。
        """
        if v is None:
            return (0, 0, 0)
        t = self.attr_type[attr]
        if t == "bool":
            return (1, 0 if not v else 1, 0)
        # int64 / text：载荷按数值或码点比较
        return (1, 0, v)
