# 因子化关系交付器（factor_delivery）

把一棵**等值连接树**的结果以因子 DAG 保存进单个 SQLite 文件，重开文件后
无需原始来源即可：

- `count()`：精确总重数（袋语义，含 multiplicity 乘积与投影合并）；
- `page(r, n)`：按输出全序读取从秩 `r` 开始的 `n` 行（秩从 0 开始，
  重复元组各占一个秩，`r` 越界返回空，`n=0` 不枚举）；
- `restricted(conds)`：在输出列上做等值 / is_null 筛选，**在结果语义上
  重新计算连续秩**，而不是从未筛选的秩处截取；
- `lineage(tuple)`：给定一条完整输出元组，返回紧凑**来源表达**，指出参与
  连接的各表实例及输入行序号，可浏览表达节点、计算来源组合数、判断指定
  实例行是否参与——全程不枚举来源组合；
- `apply_batch(changeset)`：在同一交付文件上**原子地、可重复地**应用批次
  修订（插入/撤销/重复次数修正），按批次编号去重，只重算受影响的连接键
  分区，重开文件后版本与批次记录齐全。

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

## 来源追溯（lineage）

`lineage(完整输出元组)` 返回一个 `LineageExpression`。它沿因子 DAG 中与
元组取值匹配的分组下行，只访问相关节点，**不枚举来源组合**：一条元组即使
有上亿种来源组合，表达节点数也只随来源行与因子结构增长。

```python
with DeliveryStore.open("delivery.db") as store:
    sess = QuerySession(store)
    expr = sess.lineage(("d", "a", "x"))

    expr.exists()                 # 不存在的输出元组返回 False（不报错）
    expr.combination_count()      # 来源组合数 == 该元组在结果袋中的重数
    expr.node_count()             # 表达节点数（有界，不随组合数增长）
    expr.nodes()                  # 浏览全部表达节点（DFS 去重）
    expr.source_refs()            # [(实例别名, row_seq, multiplicity), ...]
    expr.participates("l", 0)     # 实例 l 的第 0 个输入行是否参与该元组
    print(expr.to_text())         # 人类可读的来源树

    # 按实例 + 装载序号直接审计来源行（值与 mult，bool 已还原）
    sess.source_row("l", 0)       # {'row_seq':0, 'mult':1, 'values':{...}}
    sess.source_row_count("l")    # 实例 l 的来源行序号空间 0..n-1
```

### 来源表达节点的含义

来源表达是一棵 `src` / `prod` / `alt` 组成的 DAG，持久化在
`fd_prov_node` / `fd_prov_src` / `fd_prov_comp` 中。节点 `node_id > 0`
可直接在 SQLite 文件里按 id 审查；`node_id < 0` 是本次查询在内存中合成的
包装节点（对持久化节点做 prod/alt），不写入文件。

| kind | 含义 | 组合数（weight） |
|------|------|------------------|
| `src` | 某表**实例**（角色别名）上的一组来源行，rows 为 `(row_seq, mult)`；`mult` 是该行代表的重复次数，作为数据保留、不逐份建节点 | 各行 mult 之和 |
| `prod` | 连接相乘：各分支在**同一组合**中共同贡献 | 各孩子 weight 之积 |
| `alt` | 袋并相加：投影后多个**互斥**连接分支都能得到该元组 | 各孩子 weight 之和 |

关键语义保证：

1. **输入行序号按来源表装载顺序确定**（从 0 起，存于 src/red 表的
   `row_seq`）。值完全相同、序号不同的行仍是不同来源行，分别保留。
2. **同一来源表被多个实例引用时身份独立**：每个角色别名有自己的 `src`
   节点，互不混淆。
3. **multiplicity 不展开**：一行 multiplicity=N 在 `src` 节点里只是一行
   `(row_seq, N)`，来源组合数经 weight 精确计入，绝不制造 N 个节点。
4. **不同连接键分组不交叉**：投影折叠连接键时用 `alt` 保留各父键分支，
   `(a×b) ∪ (c×d)` 不会被改写成 `(a∪c)×(b∪d)`。
5. **来源组合数与结果重数一致**：导航时独立沿因子路径重算该元组重数，
   并与表达根 weight、各节点缓存 weight 交叉核对，篡改任一层都会被检出。

错误约定：输出元组列数不符、值的类型与输出列声明不一致（如 int64 列传
字符串、bool 列传整数）、向不可空列传 NULL，均抛 `QueryError`；不存在的
输出元组得到根为 `None` 的空表达（`exists()` 为 `False`，组合数 0）。
`participates` 的别名未知或序号类型非法抛 `QueryError`，序号越界返回
`False`。`count()` / `page()` / `restricted()` 的语义保持不变。

## 因子节点协议（可审查的节点表）

所有因子节点保存在 `fd_node`，明细在 `fd_group_entry` /
`fd_product_child`。每个节点都带 `weight`（缓存重数）、`scope`（作用域
逻辑属性，按输出顺序）与 `prov_id`（来源闭包，empty 节点为 NULL）。
查询时每个节点的重数都从子节点**自底向上重算**并与缓存值交叉核对，
绝不只信一个总行数字段。来源表达单独保存在 `fd_prov_node` /
`fd_prov_src` / `fd_prov_comp`，见上一节「来源表达节点的含义」。
来源表与约减表都带 `row_seq`（来源行装载序号，约减只删行不重排）。

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

## 可重复应用的批次增量更新

设备目录每周收到部件/供货点修订时，不必再对数亿种配套关系全量重算。
`factor_delivery.batch` 在同一交付文件上提供**可重复应用、原子生效、
按编号去重**的批次更新：

```python
from factor_delivery import ChangeSet, BatchChange, apply_batch

cs = ChangeSet("B-2026-W47")
cs.add(BatchChange.insert("parts", ("d-200", "密封圈"), mult=1))   # 插入
cs.add(BatchChange.retract("parts", rid=3))                        # 按 rid 撤销
cs.add(BatchChange.update_mult("parts", rid=0, new_mult=4))        # 重复次数修正
result = apply_batch(store, cs)        # 单事务一次性生效
```

关键语义与保证：

1. **稳定记录身份**：源表行有独立 `rid`（主键）与单调递增的 `row_seq`
   高水位。撤销/修正按旧 `rid` 定位；新记录即使与旧记录值完全相同也分配
   全新身份，绝不复用。撤销是软删除（`live=0`），身份保留为历史但不再
   出现于任何新版本的来源表达。
2. **多关系/多角色一次修订**：一个批次可同时改多张表；同一张物理表被
   多个角色（实例别名）引用时，修订只提交一次，所有角色同时看到。
3. **批次按编号去重**：已成功处理的 `batch_no` 再次携带**相同内容**
   （SHA-256 内容哈希一致）返回原 `BatchResult`、不产生新版本；内容不同
   抛 `BatchConflictError` 明确拒绝。批次日志（`fd_batch_log` /
   `fd_batch_change`）随事务持久化，重开文件后去重信息仍在。
4. **只处理受影响的连接键分区**：变更键沿连接边双向传播到不动点
   （严格等值，NULL 不匹配），只重算脏分区的约减行与子树因子；干净分区
   经 `fd_subtree_factor` 索引**种子化复用**其持久化因子（含来源闭包），
   绝不展开配对后的全部记录组合。`BatchResult` 给出可核验的处理范围计数：
   受影响/重建/种子化分区数、约减扫描行数、新写与回收的因子/来源节点数。
5. **来源身份延续**：干净因子的来源闭包以惰性引用（`PLazyRef`）保留旧
   `prov_id`，未改记录仍可被原 `row_seq` 引用；新因子从来源高水位分配新
   id；新版本不可达的旧因子/来源节点在同事务内回收。
6. **原子性**：输入校验、增量计算、落盘任一失败都回滚到此前版本，
   不暴露半成品；提交前进程退出（含强制 `os._exit`）后重开文件只能读到
   旧版本。成功提交后新进程重开即恢复最新版本与全部批次去重信息，无需
   重新提供最初数据集。
7. **新版本查询一致**：提交后 `count()` / `page()` / `restricted()` /
   `lineage()` 全部反映新版本，且与从修订后数据独立全量重算的结果逐项
   一致（见 `tests/test_batch_updates.py`）。

完整生命周期示例见 `examples/batch_update_lifecycle.py`；
数百万组合只修订少数键分区的演练见
`tests/demo_incremental_scale.py`。

## 校验

- `python3 -I tests/test_correctness.py`：16 个小材料，逐案与独立逐行
  枚举的完整袋及每个秩对比；
- `python3 -I tests/test_randomized.py`：随机小树 / 数据 / 投影顺序 /
  条件的属性测试；
- `python3 -I tests/test_lineage.py`：来源追溯专项。用两套独立实现互校——
  逐行枚举每个来源组合的 `brute_force_lineage`，与对紧凑表达的独立展开器
  `expand_expression`；覆盖 multiplicity、同值异序行、同表多实例身份、
  投影 alt 不交叉、多键分组隔离、空结果、类型/列错误、新进程重开等；
- `python3 -I tests/test_batch_updates.py`：批次增量专项。独立内存全量
  重算器逐批次对比完整结果与来源，覆盖多角色共用同一关系、两端同时增加、
  同内容不同身份、重复次数加减、撤销、投影后结果重合、null 键不匹配、
  先清空再恢复、批次重放与编号冲突，以及提交前故障/强制退出的原子性与
  成功后的重开恢复；
- `python3 -I tests/demo_scale.py`：20000×20000=4 亿行主演练，记录节点数、
  文件大小、SQL 访问计数，并验证除余定位、d2 隔离、投影重数相加；
- `python3 -I tests/demo_incremental_scale.py`：400 万组合只修订 1 个键
  分区的增量演练，核对总数/不同位置小页/来源查询，并用节点数与扫描计数
  证明更新不遍历整个组合结果；
- `python3 -I tests/demo_lineage_scale.py`：单条输出元组承载 **12.25 亿**
  来源组合，表达仅 9 个节点；验证节点有界、参与查询不展开（个位数 SQL）、
  多键分组 alt 不交叉、新进程重开可追溯，并用小样本独立枚举器逐组合核对。

## 明确不支持

环形连接、外连接、任意 SQL、并发发布。
（增量更新已支持，见上一节。）
