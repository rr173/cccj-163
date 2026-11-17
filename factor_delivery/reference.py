"""独立参考实现：直接逐行做嵌套循环等值连接并显式枚举完整袋。

仅用于小材料核对，绝不参与正式构建/查询路径。逻辑刻意写得直白、
与因子化实现完全独立（不共享任何节点代码）。
"""

from __future__ import annotations

from itertools import product as cartesian
from typing import Any, Dict, List, Sequence, Tuple

from .spec import Spec


def _sort_key(spec: Spec, attr: str, v: Any) -> Tuple[Any, ...]:
    if v is None:
        return (0, 0, 0)
    if spec.attr_type[attr] == "bool":
        return (1, 0 if not v else 1, 0)
    return (1, 0, v)


def brute_force_join(spec: Spec) -> List[Tuple[Tuple[Any, ...], int]]:
    """返回 [(输出元组, 合并后 multiplicity), ...]，按全序排列。"""
    spec.validate()
    root = spec.root_alias()
    parent: Dict[str, str] = {}
    parent_edge: Dict[str, Any] = {}
    stack = [root]
    seen = {root}
    order: List[str] = [root]
    while stack:
        u = stack.pop()
        for v, edge in spec.adj[u]:
            if v in seen:
                continue
            seen.add(v)
            parent[v] = u
            parent_edge[v] = edge
            stack.append(v)
            order.append(v)

    # 每个实例的 (取值元组, multiplicity)
    rows_of: Dict[str, List[Tuple[Tuple[Any, ...], int]]] = {}
    attr_pos: Dict[str, Dict[str, int]] = {}
    for alias, tname in spec.instances.items():
        tbl = spec.tables[tname]
        rows_of[alias] = list(zip(
            [tuple(r) for r in tbl.rows],
            list(tbl.multiplicities),
        ))
        attr_pos[alias] = {col.attr: i for i, col in enumerate(tbl.columns)}

    # 逐组合枚举（树连接 = 逐边嵌套）
    bag: Dict[Tuple[Any, ...], int] = {}
    choices = [rows_of[a] for a in order]
    for combo in cartesian(*choices):
        assigned: Dict[str, Tuple[Any, ...]] = {}
        weight = 1
        ok = True
        for alias, (vals, mult) in zip(order, combo):
            if alias != root:
                edge = parent_edge[alias]
                p = parent[alias]
                pvals = assigned[p]
                for cl, cr in zip(edge.cols_left, edge.cols_right):
                    if edge.alias_left == alias:
                        a_col, p_col = cl, cr
                    else:
                        a_col, p_col = cr, cl
                    a_attr = spec.tables[spec.instances[alias]].col(a_col).attr
                    p_attr = spec.tables[spec.instances[p]].col(p_col).attr
                    x = vals[attr_pos[alias][a_attr]]
                    y = pvals[attr_pos[p][p_attr]]
                    # 普通相等，null 不匹配
                    if x is None or y is None or x != y:
                        ok = False
                        break
                if not ok:
                    break
            assigned[alias] = vals
            weight *= mult
        if not ok:
            continue
        out = tuple(
            _first_value(spec, assigned, attr_pos, attr)
            for attr in spec.output_attrs
        )
        bag[out] = bag.get(out, 0) + weight

    sorted_items = sorted(
        bag.items(),
        key=lambda item: tuple(
            _sort_key(spec, attr, v)
            for attr, v in zip(spec.output_attrs, item[0])
        ),
    )
    return sorted_items


def _first_value(spec: Spec, assigned: Dict[str, Tuple[Any, ...]],
                 attr_pos: Dict[str, Dict[str, int]], attr: str) -> Any:
    # 运行交集保证所有绑定一致，取任意一个出现处即可
    for alias, tname in spec.instances.items():
        tbl = spec.tables[tname]
        if attr in attr_pos[alias]:
            return assigned[alias][attr_pos[alias][attr]]
    raise KeyError(attr)


def brute_force_count(spec: Spec) -> int:
    return sum(m for _, m in brute_force_join(spec))


def brute_force_page(spec: Spec, r: int, n: int) -> List[Tuple[Any, ...]]:
    """显式展开重复元组后按秩切片。"""
    expanded: List[Tuple[Any, ...]] = []
    for tup, mult in brute_force_join(spec):
        expanded.extend([tup] * mult)
    if r < 0 or n <= 0 or r >= len(expanded):
        return [] if n != 0 else []
    return expanded[r:r + n]


def brute_force_filtered(
    spec: Spec, conditions: Sequence[Tuple[str, str, Any]]
) -> List[Tuple[Tuple[Any, ...], int]]:
    """条件格式 (attr, 'eq', value) 或 (attr, 'is_null', None)，重新编号。"""
    items = brute_force_join(spec)
    pos = {a: i for i, a in enumerate(spec.output_attrs)}
    kept: List[Tuple[Tuple[Any, ...], int]] = []
    for tup, mult in items:
        good = True
        for attr, op, val in conditions:
            v = tup[pos[attr]]
            if op == "is_null":
                if v is not None:
                    good = False
                    break
            else:
                if v is None or v != val:
                    good = False
                    break
        if good:
            kept.append((tup, mult))
    return kept
