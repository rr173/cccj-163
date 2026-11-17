"""因子化关系交付器 (factorized relational delivery).

把一棵等值连接树的结果以因子 DAG（分组 / 笛卡尔积 / 叶子 / 空节点）保存到
SQLite，重开文件后无需原始来源即可：

* count()                 —— 精确总重数（袋语义）
* page(r, n)              —— 按全序读取第 r 秩开始的 n 行
* restricted(conds)       —— 固定部分字段（等值 / is_null）后重新编号计数、取页

整个建立与查询过程都不会逐行走过完整逻辑输出。
"""

from .spec import Column, Table, Edge, Spec, SpecError
from .builder import build_factorized, BuildError
from .store import DeliveryStore, ProtocolError
from .query import QuerySession, Condition
from .reference import (
    brute_force_join, brute_force_count, brute_force_page, brute_force_filtered,
)

__all__ = [
    "Column",
    "Table",
    "Edge",
    "Spec",
    "SpecError",
    "build_factorized",
    "BuildError",
    "DeliveryStore",
    "ProtocolError",
    "QuerySession",
    "Condition",
    "brute_force_join",
    "brute_force_count",
    "brute_force_page",
    "brute_force_filtered",
]
