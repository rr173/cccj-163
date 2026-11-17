"""来源追溯查询层：重开交付文件后，仅凭因子 DAG 与来源表达 DAG 工作。

给定一条 *完整输出元组*，沿因子 DAG 中与元组取值匹配的 group entry 下行：

* group   —— 选走取值相等的那一个 entry，其余取值分支与本元组无关；
* product —— 各分支在同一连接组合中共同贡献，来源表达做 prod（相乘）；
* union   —— 投影折叠出的袋并：只有真正产出该元组的分支入选，做 alt（相加），
             不同连接键分组的来源任选项不会被交叉相乘；
* leaf    —— 取构建期持久化的 src 来源闭包（实例别名 + 来源行序号 + mult）。

整个过程只访问与该元组相关的因子节点与来源节点，节点数随源行/因子结构
增长，**绝不枚举来源组合**：即使一条元组有上亿种来源组合，返回的表达
节点数也保持有界。来源组合数直接等于表达根的 weight。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .store import ProtocolError
from .query import QuerySession, QueryError, RNode


# --------------------------------------------------------------------- #
# 面向使用方的来源表达节点
# --------------------------------------------------------------------- #
@dataclass
class LineageNode:
    """来源表达中的一个节点。

    node_id：
        正整数表示持久化在 ``fd_prov_node`` 中的节点，可直接在 SQLite 文件里
        按 id 审查（src 行清单见 ``fd_prov_src``，复合分支见 ``fd_prov_comp``）；
        负整数表示本次查询在内存中合成的会话节点（对持久化节点做 prod/alt
        包装），不写入文件。
    kind：
        - ``"src"``：某个表实例（alias）上的来源行集合；rows 为
          (输入行序号 row_seq, 该行 multiplicity) 的升序清单。mult 是该行
          代表的重复次数，不会被逐份展开成多个节点；
        - ``"prod"``：连接相乘，children 各分支在同一组合中共同贡献，
          weight = 各孩子 weight 之积；
        - ``"alt"``：袋并相加，children 是互斥的多个来源分支（投影后不同
          连接分支得到同一输出元组），weight = 各孩子 weight 之和。
    weight：该节点覆盖的来源组合数（精确整数，不展开即可得到）。
    """

    node_id: int
    kind: str
    weight: int
    alias: Optional[str] = None
    rows: Optional[List[Tuple[int, int]]] = None
    children: List["LineageNode"] = field(default_factory=list)

    @property
    def persisted(self) -> bool:
        return self.node_id > 0

    def describe(self) -> str:
        """单行人类可读描述。"""
        if self.kind == "src":
            seqs = ", ".join(f"#{s}(×{m})" for s, m in (self.rows or []))
            return f"src 实例 {self.alias!r} 的来源行 [{seqs}]"
        op = "相乘(prod)" if self.kind == "prod" else "袋并(alt)"
        where = "持久化" if self.persisted else "会话内"
        return f"{self.kind} {op}，{where}节点，{len(self.children)} 个分支"


class LineageExpression:
    """一条输出元组的紧凑来源表达（不存在的元组根为 None，组合数为 0）。"""

    def __init__(self, session: QuerySession, tuple_value: Tuple[Any, ...],
                 root: Optional[LineageNode]):
        self._session = session
        self.tuple_value = tuple_value
        self.root = root

    # ---------------- 基本性质 ---------------- #
    def exists(self) -> bool:
        """该输出元组是否存在于结果袋中。"""
        return self.root is not None

    def combination_count(self) -> int:
        """产生该元组的来源组合数；等于该元组在结果袋中的重数。

        直接读取表达根 weight，不枚举任何组合。
        """
        return self.root.weight if self.root is not None else 0

    @property
    def root_id(self) -> Optional[int]:
        return self.root.node_id if self.root is not None else None

    # ---------------- 节点浏览 ---------------- #
    def nodes(self) -> List[LineageNode]:
        """表达中的全部节点（按 DFS 去重，共享子表达只出现一次）。"""
        if self.root is None:
            return []
        out: List[LineageNode] = []
        seen: set = set()

        def walk(n: LineageNode) -> None:
            if n.node_id in seen:
                return
            seen.add(n.node_id)
            out.append(n)
            for c in n.children:
                walk(c)

        walk(self.root)
        return out

    def node_count(self) -> int:
        return len(self.nodes())

    def persisted_node_count(self) -> int:
        return sum(1 for n in self.nodes() if n.persisted)

    def virtual_node_count(self) -> int:
        return sum(1 for n in self.nodes() if not n.persisted)

    def source_refs(self) -> List[Tuple[str, int, int]]:
        """展开到 src 叶子层面的引用清单：(实例别名, row_seq, multiplicity)。

        注意这只列出 *不同的来源行*（每行一条，mult 随行给出），不是来源
        组合；数量随来源行数增长，与组合数无关。
        """
        out: List[Tuple[str, int, int]] = []
        for n in self.nodes():
            if n.kind == "src":
                for seq, mult in n.rows or []:
                    out.append((n.alias, seq, mult))  # type: ignore[arg-type]
        out.sort()
        return out

    # ---------------- 参与判断 ---------------- #
    def participates(self, alias: str, row_seq: int) -> bool:
        """指定实例的指定输入行是否参与了该输出元组（任一来源组合即可）。

        沿表达 DAG 走到 src 叶子判断，不枚举来源组合。别名未知或序号类型
        非法时报 QueryError；序号越界（该行不存在）返回 False。
        """
        instances = self._session.store.instances
        if not isinstance(alias, str) or alias not in instances:
            raise QueryError(
                f"未知表实例 {alias!r}，已知实例: {sorted(instances)}"
            )
        if not isinstance(row_seq, int) or isinstance(row_seq, bool) or row_seq < 0:
            raise QueryError("row_seq 必须是非负整数")
        # 越界按 row_seq 高水位判断（含已撤销记录的序号空洞），不能用
        # 存活行数：撤销后新记录的合法序号可能大于存活行数。
        if row_seq >= self._session.store.row_seq_high_watermark(alias):
            return False
        if self.root is None:
            return False

        def hit(n: LineageNode) -> bool:
            if n.kind == "src":
                return n.alias == alias and \
                    any(s == row_seq for s, _ in (n.rows or []))
            return any(hit(c) for c in n.children)

        return hit(self.root)

    # ---------------- 渲染 ---------------- #
    def to_text(self) -> str:
        if self.root is None:
            return f"输出元组 {self.tuple_value!r} 不存在，来源表达为空。"
        lines = [
            f"输出元组 {self.tuple_value!r} 的来源表达",
            f"来源组合数（= 结果重数）: {self.combination_count()}",
            f"表达节点数: {self.node_count()}（持久化 "
            f"{self.persisted_node_count()} / 会话内 {self.virtual_node_count()}）",
            "节点树（id>0 可在 fd_prov_node/fd_prov_src/fd_prov_comp 审查）:",
        ]

        def render(n: LineageNode, depth: int, last: bool, prefix: str) -> None:
            connector = "└─ " if last else "├─ "
            head = f"{prefix}{connector if depth else ''}"
            lines.append(f"{head}[{n.node_id}] {n.describe()} "
                         f"weight={n.weight}")
            child_prefix = prefix + ("   " if last else "│  ")
            for i, c in enumerate(n.children):
                render(c, depth + 1, i == len(n.children) - 1, child_prefix)

        render(self.root, 0, True, "")
        return "\n".join(lines)


# --------------------------------------------------------------------- #
# 导航器
# --------------------------------------------------------------------- #
class _LineageNavigator:
    def __init__(self, session: QuerySession):
        self.session = session
        self.conn = session.conn
        self._prov_cache: Dict[int, LineageNode] = {}
        self._nav_cache: Dict[Tuple[int, int], Optional[LineageNode]] = {}
        self._virtual_seq = 0

    def lineage_of(self, tuple_value: Tuple[Any, ...]) -> Optional[LineageNode]:
        root = self.session.root
        if root.kind == "empty":
            return None
        node = self._navigate(root, tuple_value, 0)
        if node is None:
            return None
        if node.weight != self._tuple_factor_weight(root, tuple_value):
            raise ProtocolError("来源表达组合数与因子侧路径重数不符")
        return node

    # ---------------- 因子 DAG 导航 ---------------- #
    def _navigate(self, node: RNode, values: Tuple[Any, ...],
                  pos: int) -> Optional[LineageNode]:
        key = (node.node_id, pos)
        if key in self._nav_cache:
            return self._nav_cache[key]
        node._ensure()
        # 顺带核对持久化来源闭包与因子节点重数一致（不只信总行数字段）
        self._check_closure_weight(node)

        result: Optional[LineageNode]
        if node.kind == "empty":
            result = None
        elif node.kind == "leaf":
            result = self._load_leaf_prov(node)
        elif node.kind == "group":
            result = self._navigate_group(node, values, pos)
        elif node.kind == "product":
            result = self._navigate_product(node, values, pos)
        elif node.kind == "union":
            result = self._navigate_union(node, values, pos)
        else:
            raise ProtocolError(f"未知因子节点类型 {node.kind}")

        if result is not None and result.weight <= 0:
            raise ProtocolError("来源表达出现非正组合数")
        self._nav_cache[key] = result
        return result

    def _navigate_group(self, node: RNode, values: Tuple[Any, ...],
                        pos: int) -> Optional[LineageNode]:
        attr = node._group_attr()
        want = values[pos]
        for value, cid in node._entries:  # type: ignore[union-attr]
            if self._values_equal(attr, value, want):
                return self._navigate(self.session.get_node(cid),
                                      values, pos + 1)
        return None

    def _navigate_product(self, node: RNode, values: Tuple[Any, ...],
                          pos: int) -> Optional[LineageNode]:
        branches: List[LineageNode] = []
        cur = pos
        for cid in node._child_ids:  # type: ignore[union-attr]
            child = self.session.get_node(cid)
            sub = self._navigate(child, values, cur)
            if sub is None:
                return None
            branches.append(sub)
            cur += len(child.scope)
        return self._make_composite("prod", branches)

    def _navigate_union(self, node: RNode, values: Tuple[Any, ...],
                        pos: int) -> Optional[LineageNode]:
        # 袋并各分支同作用域、用同一段输出位置分别导航；只有产出该元组的
        # 分支入选。这里是 alt（相加），绝不把分支重新做成乘积，
        # 因此不同连接键分组的来源不会交叉组合。
        branches: List[LineageNode] = []
        for cid in node._child_ids:  # type: ignore[union-attr]
            sub = self._navigate(self.session.get_node(cid), values, pos)
            if sub is not None:
                branches.append(sub)
        if not branches:
            return None
        return self._make_composite("alt", branches)

    # ---------------- 持久化来源节点加载 ---------------- #
    def _load_leaf_prov(self, leaf: RNode) -> LineageNode:
        # 叶子的持久化闭包通常是 src；当一个 product 的所有分支都是空作用域
        # 叶子（无输出列的连接）时会折叠成带 prod/alt 闭包的叶子，同样合法。
        prov_id = self._prov_id_of(leaf)
        node = self._load_prov(prov_id)
        if node.weight != leaf.weight:
            raise ProtocolError(
                f"叶子 {leaf.node_id} 来源重数 {node.weight} 与因子重数 "
                f"{leaf.weight} 不符"
            )
        return node

    def _load_prov(self, prov_id: int) -> LineageNode:
        cached = self._prov_cache.get(prov_id)
        if cached is not None:
            return cached
        row = self.conn.execute(
            "SELECT kind, weight FROM fd_prov_node WHERE prov_id=?",
            (prov_id,),
        ).fetchone()
        if row is None:
            raise ProtocolError(f"来源节点 {prov_id} 缺失")
        kind, weight_text = row
        weight = int(weight_text)
        if kind == "src":
            rows = self.conn.execute(
                "SELECT alias, row_seq, mult FROM fd_prov_src "
                "WHERE prov_id=? ORDER BY position",
                (prov_id,),
            ).fetchall()
            if not rows:
                raise ProtocolError(f"src 来源节点 {prov_id} 没有任何来源行")
            alias = rows[0][0]
            node = LineageNode(
                prov_id, "src", weight,
                alias=alias,
                rows=[(r[1], r[2]) for r in rows],
            )
        else:
            child_ids = [r[0] for r in self.conn.execute(
                "SELECT child_prov FROM fd_prov_comp "
                "WHERE prov_id=? ORDER BY position",
                (prov_id,),
            ).fetchall()]
            children = [self._load_prov(cid) for cid in child_ids]
            node = LineageNode(prov_id, kind, weight, children=children)
        self._verify_weight(node)
        self._prov_cache[prov_id] = node
        return node

    def _make_composite(self, kind: str,
                        branches: List[LineageNode]) -> LineageNode:
        if len(branches) == 1:
            return branches[0]
        self._virtual_seq -= 1
        if kind == "prod":
            weight = 1
            for b in branches:
                weight *= b.weight
        else:
            weight = sum(b.weight for b in branches)
        return LineageNode(self._virtual_seq, kind, weight, children=branches)

    # ---------------- 校验工具 ---------------- #
    def _prov_id_of(self, node: RNode) -> int:
        row = self.conn.execute(
            "SELECT prov_id FROM fd_node WHERE node_id=?", (node.node_id,)
        ).fetchone()
        if row is None or row[0] is None:
            raise ProtocolError(
                f"因子节点 {node.node_id} 缺少持久化来源闭包"
            )
        return int(row[0])

    def _check_closure_weight(self, node: RNode) -> None:
        row = self.conn.execute(
            "SELECT prov_id FROM fd_node WHERE node_id=?", (node.node_id,)
        ).fetchone()
        if row is None or row[0] is None:
            if node.kind != "empty":
                raise ProtocolError(
                    f"因子节点 {node.node_id} 缺少持久化来源闭包"
                )
            return
        prov = self._load_prov(int(row[0]))
        if prov.weight != node.weight:
            raise ProtocolError(
                f"因子节点 {node.node_id} 的来源闭包重数 {prov.weight} "
                f"与因子重数 {node.weight} 不符"
            )

    def _verify_weight(self, node: LineageNode) -> None:
        if node.kind == "src":
            total = sum(m for _, m in node.rows or [])
        elif node.kind == "prod":
            total = 1
            for c in node.children:
                total *= c.weight
        else:
            total = sum(c.weight for c in node.children)
        if total != node.weight:
            raise ProtocolError(
                f"来源节点 {node.node_id} 缓存组合数 {node.weight} 与按孩子"
                f"重算值 {total} 不符"
            )

    def _tuple_factor_weight(self, root: RNode,
                             values: Tuple[Any, ...]) -> int:
        """独立沿因子路径重算该元组的重数，与表达根交叉核对。"""
        def weight_of(node: RNode, pos: int) -> int:
            node._ensure()
            if node.kind == "empty":
                return 0
            if node.kind == "leaf":
                return node.weight
            if node.kind == "group":
                attr = node._group_attr()
                for value, cid in node._entries:  # type: ignore[union-attr]
                    if self._values_equal(attr, value, values[pos]):
                        return weight_of(self.session.get_node(cid), pos + 1)
                return 0
            if node.kind == "product":
                total = 1
                for cid in node._child_ids:  # type: ignore[union-attr]
                    child = self.session.get_node(cid)
                    total *= weight_of(child, pos)
                    pos += len(child.scope)
                return total
            # union：仅匹配分支相加
            total = 0
            for cid in node._child_ids:  # type: ignore[union-attr]
                total += weight_of(self.session.get_node(cid), pos)
            return total

        return weight_of(root, 0)

    def _values_equal(self, attr: str, disk_value: Any,
                      query_value: Any) -> bool:
        """按属性声明类型严格判等（bool 与 int 不混淆，NULL 只等于 NULL）。"""
        if disk_value is None or query_value is None:
            return disk_value is None and query_value is None
        dtype = self.session.dtype_of(attr)
        if dtype == "bool":
            return isinstance(query_value, bool) and disk_value == query_value
        if dtype == "int64":
            return isinstance(query_value, int) \
                and not isinstance(query_value, bool) \
                and disk_value == query_value
        return disk_value == query_value


# --------------------------------------------------------------------- #
# 元组校验与入口
# --------------------------------------------------------------------- #
def validate_output_tuple(session: QuerySession,
                          tuple_value: Any) -> Tuple[Any, ...]:
    """校验完整输出元组：列数一致、类型与输出列声明一致、NULL 合法。"""
    if not isinstance(tuple_value, (tuple, list)):
        raise QueryError(
            f"输出元组必须是 tuple 或 list，得到 {type(tuple_value).__name__}"
        )
    attrs = session.output_attrs
    if len(tuple_value) != len(attrs):
        raise QueryError(
            f"输出元组应有 {len(attrs)} 列（{attrs}），实际得到 "
            f"{len(tuple_value)} 个值"
        )
    out: List[Any] = []
    for attr, value in zip(attrs, tuple_value):
        dtype, nullable = session.store.attr_meta(attr)
        if value is None:
            if not nullable:
                raise QueryError(f"输出列 {attr!r} 不可为空，得到 NULL")
            out.append(None)
            continue
        if dtype == "int64":
            if isinstance(value, bool) or not isinstance(value, int):
                raise QueryError(
                    f"输出列 {attr!r} 需要 int64，得到 "
                    f"{type(value).__name__}: {value!r}"
                )
        elif dtype == "text":
            if not isinstance(value, str):
                raise QueryError(
                    f"输出列 {attr!r} 需要 text，得到 "
                    f"{type(value).__name__}: {value!r}"
                )
        else:  # bool
            if not isinstance(value, bool):
                raise QueryError(
                    f"输出列 {attr!r} 需要 bool，得到 "
                    f"{type(value).__name__}: {value!r}"
                )
        out.append(value)
    return tuple(out)


def lineage_of(session: QuerySession, tuple_value: Any) -> LineageExpression:
    """返回一条完整输出元组的来源表达；元组不存在时根为 None（不报错）。

    类型错误、列数不符、向不可空列传 NULL 等情况抛 QueryError。
    """
    values = validate_output_tuple(session, tuple_value)
    navigator = _LineageNavigator(session)
    root = navigator.lineage_of(values)
    return LineageExpression(session, values, root)
