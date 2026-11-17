# 因子化关系交付器（factor_delivery）

把一棵**等值连接树**的结果以因子 DAG 保存进单个 SQLite 文件，重开文件后
无需原始来源即可：

- `count()`：精确总重数（袋语义，含 multiplicity 乘积与投影合并）；
- `page(r, n)`：按输出全序读取从秩 `r` 开始的 `n` 行（秩从 0 开始，
  重复元组各占一个秩，`r` 越界返回空，`n=0` 不枚举）；
- `restricted(conds)`：在输出列上做等值 / is_null 筛选，**在结果语义上
  重新计算连续秩**，而不是从未筛选的秩处截取。

建立结构与查询都**不会逐行走过完整逻辑输出**：4 亿组合行只产生随源表 /
键分组增长的节点，文件里没有宽连接结果。

## 快速使用

```python
from factor_delivery import (
    Column, Table, Edge, Spec,
    build_factorized, DeliveryStore, QuerySession, Condition,
)

spec = Spec(
    tables={"device": device, "left": left, "right": right},
    instances={"d": "device", "l": "left", "r": "right"},
    edges=[
        Edge("d", ("id",), "l", ("dev",)),
        Edge("d", ("id",), "r", ("dev",)),
    ],
    output_attrs=["device", "part", "supply"],
)
store = DeliveryStore.create("delivery.db", spec)
build_factorized(store, spec)          # 半连接约减 + 因子 DAG
store.close()

# 重开：只依赖交付文件，不依赖源表 / 源库
with DeliveryStore.open("delivery.db") as store:
    sess = QuerySession(store)
    sess.count()                       # 精确总重数
    sess.page(0, 10)                   # 全序取页
    v = sess.restricted([Condition.eq("device", "d")])
    v.count()                          # 条件后重新编号的重数
    v.page(0, 10)
```

## 因子节点协议（可审查的节点表）

所有因子节点保存在 `fd_node`，明细在 `fd_group_entry` /
`fd_product_child`。每个节点都带 `weight`（缓存重数）与 `scope`（作用域
逻辑属性，按输出顺序）。查询时每个节点的重数都从子节点**自底向上重算**
并与缓存值交叉核对，绝不只信一个总行数字段。

| kind    | 行域（domain）                          | 计数（weight）                     | 组合意义 |
|---------|----------------------------------------|------------------------------------|----------|
| group   | 某逻辑属性的排序取值域，每值一个孩子     | 各 entry 孩子重数之和              | 在该属性上分组；NULL 最先 |
| product | 多个**作用域不重叠**、条件独立的分支     | 各分支重数之积                     | 仅在共享连接键固定后才出现的笛卡尔积 |
| union   | 同作用域、互不条件独立的多个分支         | 各分支重数之和（袋并）             | 投影折叠连接键后保留各父键分支，杜绝跨键笛卡尔积 |
| leaf    | 单一（已固定的）输出元组                | 合并后的 multiplicity              | 袋的最小单元 |
| empty   | 空                                     | 0                                  | 被约减消除的无效分支 |

关键依赖不变量：

1. **product 只组合不重叠输出属性**，且仅在父连接键已固定、分支条件独立时
   构造；不同父键下的候选永远不会被直接相乘。
2. **union 保留袋并**：投影消掉连接键后，`(a×b) ∪ (c×d)` 不会被错误改写成
   `(a∪c)×(b∪d)`，相同输出元组的重数在 group 边界相加。
3. 共享逻辑属性在全树只有一次一致绑定；普通相等遇 NULL 不匹配
   （SQL `=` 语义，NULL 不等于 NULL）。

## 全序

输出列按 `output_attrs` 的定义顺序字典排序；单列内 NULL 先于非 NULL，
非 NULL 按 int64 数值、text 码点、bool 的 false<true。排序一律按属性
*声明类型* 分派（bool 经 SQLite 整数列往返后运行时是 int，不能用
`isinstance(x, bool)` 判断）。

## 两阶段构建

1. **Yannakakis 式半连接约减**：每个实例有独立物理表 `red_<alias>`，
   自叶向根、再自根向叶各做一遍 `NOT EXISTS` 半连接，消除无效 / 空分支；
   多列键走多列索引。
2. **因子装配**：按「输出可见属性 + 子边连接键」做一次性 Trie 分组，
   子树因子按 `(实例, 父键值)` 记忆化复用；再投影折叠非输出键为 union，
   最后按输出列顺序规范化 group 嵌套。无法在不展开组合的前提下满足的
   输出顺序会抛 `BuildError`。

## 校验

- `python3 -I tests/test_correctness.py`：16 个小材料，逐案与独立逐行
  枚举的完整袋及每个秩对比；
- `python3 -I tests/test_randomized.py`：随机小树 / 数据 / 投影顺序 /
  条件的属性测试；
- `python3 -I tests/demo_scale.py`：20000×20000=4 亿行主演练，记录节点数、
  文件大小、SQL 访问计数，并验证除余定位、d2 隔离、投影重数相加。

## 明确不支持

环形连接、外连接、任意 SQL、增量更新、并发发布。
