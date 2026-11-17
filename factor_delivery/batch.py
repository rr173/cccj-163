"""可重复应用的批次增量更新。

一个批次（ChangeSet）可对多张来源表同时做三类修订：

* insert  —— 插入新记录，分配全新 rid 与全新 row_seq（高水位单调递增，
  即使值与某条旧记录完全相同，也绝不复用其身份）；
* retract —— 按 rid 撤销旧记录（软删除 live=0，身份保留但不再出现于
  任何新版本的来源表达）；
* update  —— 按 rid 修正重复次数 mult（语义上等价于同身份的 mult 覆写）。

增量计算只触碰受影响的连接键分区及其压缩节点：

1. 从变更记录的连接键值出发，沿连接边双向传播脏键集合（fixed point）；
2. 约减表只重算脏分区（scoped 半连接），NULL 键天然不匹配；
3. 因子 DAG 中干净分区的子树因子经 fd_subtree_factor 索引 *种子化* 复用，
   只重建脏分区因子；
4. 来源闭包：干净因子保留旧 prov_id（PLazyRef），新因子从 prov 高水位
   分配新 id；撤销记录不再被任何新来源表达引用。

整个批次在单个 SQLite 事务内一次性生效：输入校验、增量计算、落盘任一
失败都回滚到此前版本。批次按 batch_no 去重，重放同号同内容返回原结果，
同号不同内容抛 BatchConflictError 明确拒绝。
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .spec import Edge, Spec, INT64_MIN, INT64_MAX
from .store import DeliveryStore
from .builder import (
    BuildError, BNode, BEmpty, BLeaf, BGroup, BProduct, BUnion,
    normalize, _Builder,
)
from .provenance import PNode, PLazyRef


class BatchError(ValueError):
    """批次输入校验失败（未知表/rid、值类型不符、批次为空等）。"""


class BatchConflictError(BatchError):
    """批次编号已成功处理过，但本次携带内容与原内容不同。"""


# --------------------------------------------------------------------- #
# 批次输入 / 输出
# --------------------------------------------------------------------- #
@dataclass(frozen=True)
class BatchChange:
    """单条修订。

    op:        'insert' | 'retract' | 'update'
    table:     来源表名（物理表，不是角色别名；同表多角色自动同时生效）
    rid:       insert 时忽略（由系统分配）；retract/update 必须给出旧 rid
    values:    insert 时的列值元组（按来源表列顺序）
    mult:      insert 的重复次数（正整数）；update 的新 mult；retract 忽略
    """

    op: str
    table: str
    rid: Optional[int] = None
    values: Optional[Tuple[Any, ...]] = None
    mult: Optional[int] = None

    @staticmethod
    def insert(table: str, values: Sequence[Any], mult: int = 1) -> "BatchChange":
        return BatchChange("insert", table, None, tuple(values), mult)

    @staticmethod
    def retract(table: str, rid: int) -> "BatchChange":
        return BatchChange("retract", table, rid, None, None)

    @staticmethod
    def update_mult(table: str, rid: int, new_mult: int) -> "BatchChange":
        return BatchChange("update", table, rid, None, new_mult)

    def payload(self) -> Dict[str, Any]:
        return {"op": self.op, "table": self.table, "rid": self.rid,
                "values": list(self.values) if self.values is not None else None,
                "mult": self.mult}


@dataclass
class ChangeSet:
    """一个批次：唯一编号 + 若干张表的修订。"""

    batch_no: str
    changes: List[BatchChange] = field(default_factory=list)

    def add(self, change: BatchChange) -> "ChangeSet":
        self.changes.append(change)
        return self

    def content_hash(self) -> str:
        """批次全部内容的稳定哈希（不含 batch_no 之外的易变字段）。"""
        canon = json.dumps(
            [c.payload() for c in self.changes],
            ensure_ascii=False, sort_keys=True, separators=(",", ":"),
            default=_json_default,
        )
        return hashlib.sha256(canon.encode("utf-8")).hexdigest()


def _json_default(obj: Any) -> Any:
    if isinstance(obj, tuple):
        return list(obj)
    raise TypeError(f"不可序列化的对象: {type(obj).__name__}")


@dataclass
class BatchResult:
    """批次成功处理后的可核验结果。"""

    batch_no: str
    applied_seq: int
    version: int
    total_weight: int
    content_hash: str
    # 处理范围计数（核验没有遍历整个组合结果）
    affected_partitions: Dict[str, int]
    rebuilt_partitions: Dict[str, int]
    seeded_partitions: Dict[str, int]
    red_rows_scanned: int
    red_rows_touched: int
    factor_nodes_written: int
    prov_nodes_written: int
    prov_nodes_reused: int
    orphan_factor_nodes: int
    orphan_prov_nodes: int
    inserted_rid: Dict[int, Tuple[str, int]]       # 变更位置 -> (表, 新rid)
    retracted_rids: List[Tuple[str, int]]
    updated_rids: List[Tuple[str, int]]
    stats: Dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(
            {k: v for k, v in self.__dict__.items()},
            ensure_ascii=False, sort_keys=True, default=_json_default,
        )

    @staticmethod
    def from_json(text: str) -> "BatchResult":
        data = json.loads(text)
        return BatchResult(**data)


# --------------------------------------------------------------------- #
# 运行时规格：重开文件后仅凭协议表重建 builder 所需的结构信息
# （不再需要最初的数据集）
# --------------------------------------------------------------------- #
class _RuntimeColumn:
    def __init__(self, name: str, dtype: str, nullable: bool, attr: str):
        self.name = name
        self.dtype = dtype
        self.nullable = nullable
        self.attr = attr


class _RuntimeTable:
    def __init__(self, name: str, columns: List[_RuntimeColumn]):
        self.name = name
        self.columns = columns
        self._col_by_name = {c.name: c for c in columns}

    def col(self, name: str) -> _RuntimeColumn:
        return self._col_by_name[name]


class RuntimeSpec:
    """从交付文件协议表恢复的连接结构（无数据行），接口与 Spec 对齐。"""

    def __init__(self, store: DeliveryStore):
        self.store = store
        self.tables: Dict[str, _RuntimeTable] = {}
        for tname in store.source_table_names():
            rows = store.conn.execute(
                "SELECT col_name, dtype, nullable, attr FROM fd_source_column "
                "WHERE table_name=? ORDER BY position",
                (tname,),
            ).fetchall()
            cols = [_RuntimeColumn(n, t, bool(nul), a) for n, t, nul, a in rows]
            self.tables[tname] = _RuntimeTable(tname, cols)
        self.instances: Dict[str, str] = dict(store.instances)
        # 边：从 fd_edge 还原
        edge_rows = store.conn.execute(
            "SELECT DISTINCT edge_id FROM fd_edge ORDER BY edge_id"
        ).fetchall()
        self.edges: List[Edge] = []
        for (eid,) in edge_rows:
            left = store.conn.execute(
                "SELECT endpoint_alias, col_name FROM fd_edge "
                "WHERE edge_id=? AND side='L' ORDER BY position", (eid,)
            ).fetchall()
            right = store.conn.execute(
                "SELECT endpoint_alias, col_name FROM fd_edge "
                "WHERE edge_id=? AND side='R' ORDER BY position", (eid,)
            ).fetchall()
            self.edges.append(Edge(
                left[0][0], tuple(c for _, c in left),
                right[0][0], tuple(c for _, c in right),
            ))
        out_rows = store.conn.execute(
            "SELECT attr, dtype, nullable FROM fd_output_attr ORDER BY position"
        ).fetchall()
        self.output_attrs: List[str] = [a for a, _, _ in out_rows]
        # 完整逻辑属性类型/可空信息从所有来源列恢复（含不输出的内部连接键），
        # 全序排序键与值校验都可能引用非输出属性。
        self.attr_type: Dict[str, str] = {}
        self.attr_nullable: Dict[str, bool] = {}
        for tname in self.tables:
            for col in self.tables[tname].columns:
                if col.attr in self.attr_type:
                    self.attr_nullable[col.attr] = \
                        self.attr_nullable[col.attr] or col.nullable
                else:
                    self.attr_type[col.attr] = col.dtype
                    self.attr_nullable[col.attr] = col.nullable
        # 邻接表与派生结构
        self.adj: Dict[str, List[Tuple[str, Edge]]] = {a: [] for a in self.instances}
        for e in self.edges:
            a, b = e.endpoints()
            self.adj[a].append((b, e))
            self.adj[b].append((a, e))
        self._orient()

    def _orient(self) -> None:
        root = self.root_alias()
        self.root_alias_name = root
        self.parent_edge: Dict[str, Optional[Edge]] = {root: None}
        self.children: Dict[str, List[str]] = {a: [] for a in self.instances}
        self.child_edge: Dict[Tuple[str, str], Edge] = {}
        stack = [root]
        seen = {root}
        while stack:
            u = stack.pop()
            for v, edge in self.adj[u]:
                if v in seen:
                    continue
                seen.add(v)
                self.parent_edge[v] = edge
                self.children[u].append(v)
                self.child_edge[(u, v)] = edge
                stack.append(v)
        self.postorder: List[str] = []
        self._post(root)

    def _post(self, u: str) -> None:
        for v in self.children[u]:
            self._post(v)
        self.postorder.append(u)

    def root_alias(self) -> str:
        return next(iter(sorted(self.instances)))

    def instance_columns(self, alias: str) -> List[_RuntimeColumn]:
        return list(self.tables[self.instances[alias]].columns)

    def edge_attrs(self, e: Edge) -> List[str]:
        ta = self.tables[self.instances[e.alias_left]]
        return [ta.col(c).attr for c in e.cols_left]

    def value_key(self, attr: str, v: Any) -> Tuple[Any, ...]:
        if v is None:
            return (0, 0, 0)
        t = self.attr_type[attr]
        if t == "bool":
            return (1, 0 if not v else 1, 0)
        return (1, 0, v)

    def validate(self) -> None:
        pass


# --------------------------------------------------------------------- #
# 因子节点磁盘加载器：把干净分区的持久化因子种子化为 BNode，
# 来源闭包用 PLazyRef 惰性引用，不展开旧来源 DAG。
# --------------------------------------------------------------------- #
class FactorLoader:
    def __init__(self, store: DeliveryStore, spec: RuntimeSpec):
        self.store = store
        self.conn = store.conn
        self.spec = spec
        self._node_cache: Dict[int, BNode] = {}
        self._prov_cache: Dict[int, PLazyRef] = {}
        self.prov_reused = 0

    def load_factor(self, node_id: int) -> BNode:
        cached = self._node_cache.get(node_id)
        if cached is not None:
            return cached
        kind, weight_text, scope_text = self.conn.execute(
            "SELECT kind, weight, scope FROM fd_node WHERE node_id=?",
            (node_id,),
        ).fetchone()
        weight = int(weight_text)
        scope = tuple(s for s in scope_text.split(",") if s)
        if kind == "empty":
            node: BNode = BEmpty(scope)
        elif kind == "leaf":
            node = BLeaf(weight, self._load_prov_ref(node_id, weight))
        elif kind == "group":
            attr = self.conn.execute(
                "SELECT attr FROM fd_group_entry WHERE node_id=? LIMIT 1",
                (node_id,),
            ).fetchone()[0]
            entries = []
            for is_null, v_int, v_text, v_bool, cid, _attr in self.conn.execute(
                "SELECT is_null, v_int, v_text, v_bool, child_id, attr "
                "FROM fd_group_entry WHERE node_id=? ORDER BY ord",
                (node_id,),
            ).fetchall():
                if is_null:
                    value: Any = None
                else:
                    dtype = self.spec.attr_type[attr]
                    if dtype == "bool":
                        value = bool(v_bool)
                    elif dtype == "int64":
                        value = v_int
                    else:
                        value = v_text
                entries.append((value, self.load_factor(cid)))
            # BGroup 会重新排序并核对 weight；磁盘顺序本就是全序
            node = BGroup(attr, entries, self.spec.value_key)
            if node.weight != weight:
                raise BuildError(
                    f"种子化 group {node_id} 重算 {node.weight} != 缓存 {weight}"
                )
        elif kind in ("product", "union"):
            cids = [r[0] for r in self.conn.execute(
                "SELECT child_id FROM fd_product_child WHERE node_id=? "
                "ORDER BY position", (node_id,),
            ).fetchall()]
            children = [self.load_factor(c) for c in cids]
            if kind == "product":
                node = BProduct(children)
            else:
                node = BUnion.__new__(BUnion)
                BNode.__init__(node, weight=weight, scope=scope, kind="union")
                node.children = children
            if node.weight != weight:
                raise BuildError(
                    f"种子化 {kind} {node_id} 重算 {node.weight} != 缓存 {weight}"
                )
        else:
            raise BuildError(f"未知磁盘因子类型 {kind}")
        # 非空复合节点挂上惰性来源引用
        if kind in ("group", "product", "union"):
            node.prov = self._load_prov_ref(node_id, weight)
        self._node_cache[node_id] = node
        return node

    def _load_prov_ref(self, factor_node_id: int, weight: int) -> PNode:
        row = self.conn.execute(
            "SELECT prov_id FROM fd_node WHERE node_id=?", (factor_node_id,)
        ).fetchone()
        if row is None or row[0] is None:
            raise BuildError(f"因子 {factor_node_id} 缺少来源闭包，无法种子化")
        pid = int(row[0])
        ref = self._prov_cache.get(pid)
        if ref is None:
            prow = self.conn.execute(
                "SELECT weight FROM fd_prov_node WHERE prov_id=?", (pid,)
            ).fetchone()
            if prow is None:
                raise BuildError(f"来源节点 {pid} 缺失")
            ref = PLazyRef(pid, int(prow[0]))
            self._prov_cache[pid] = ref
            self.prov_reused += 1
        if ref.weight != weight:
            raise BuildError(
                f"因子 {factor_node_id} 来源重数 {ref.weight} 与因子重数 {weight} 不符"
            )
        return ref


# --------------------------------------------------------------------- #
# 增量引擎
# --------------------------------------------------------------------- #
class _ScopedBuilder(_Builder):
    """带分区种子化的构建器：干净分区直接复用旧持久化因子。"""

    def __init__(self, store: DeliveryStore, spec: RuntimeSpec,
                 engine: "IncrementalEngine"):
        super().__init__(store, spec)
        self.engine = engine
        # 与全量构建相同的全序键安装（builder 模块的全局排序函数）
        from . import builder as _bmod
        _bmod._KEY_FN = spec.value_key

    def build_subtree(self, alias: str,
                      parent_key: Optional[Tuple[Any, ...]],
                      bound: Tuple[str, ...] = ()) -> BNode:
        if parent_key is not None:
            memo_key = (alias, parent_key)
            cached = self.memo.get(memo_key)
            if cached is not None:
                return cached
            if parent_key not in self.engine.dirty_partitions.get(alias, set()):
                old_id = self.store.read_subtree_factor_id(alias, parent_key)
                if old_id is not None:
                    seeded = self.engine.loader.load_factor(old_id)
                    seeded.persisted_id = old_id
                    self.memo[memo_key] = seeded
                    self.subtree_roots.append((alias, parent_key, seeded))
                    self.engine.seeded_count[alias] += 1
                    return seeded
                # 旧索引缺失（新增分区）：落到正常重建路径
        node = super().build_subtree(alias, parent_key, bound)
        if parent_key is not None:
            self.engine.rebuilt_count[alias] += 1
        return node


class IncrementalEngine:
    def __init__(self, store: DeliveryStore):
        self.store = store
        self.spec = RuntimeSpec(store)
        self.dirty_partitions: Dict[str, set] = {
            a: set() for a in self.spec.instances
        }
        # 每条边两端各自的脏键集合（_propagate_dirty 填充）
        self._edge_id = {e: i for i, e in enumerate(self.spec.edges)}
        self.dirty_edge: Dict[Tuple[int, str], set] = {}
        self.seeded_count: Dict[str, int] = {a: 0 for a in self.spec.instances}
        self.rebuilt_count: Dict[str, int] = {
            a: 0 for a in self.spec.instances
        }
        self.red_rows_scanned = 0
        self.red_rows_touched = 0
        self.loader: Optional[FactorLoader] = None
        self._old_red_phys: Dict[str, str] = {}

    # ---------------- 入口 ---------------- #
    def apply(self, changeset: ChangeSet,
              crash: Optional[str] = None) -> BatchResult:
        store = self.store
        batch_no = changeset.batch_no
        if not isinstance(batch_no, str) or not batch_no:
            raise BatchError("batch_no 必须是非空字符串")
        content_hash = changeset.content_hash()

        # 批次去重：已成功处理的编号，同内容返回原结果，异内容明确拒绝
        existing = store.lookup_batch(batch_no)
        if existing is not None:
            old_hash, _seq, old_result = existing
            if old_hash != content_hash:
                raise BatchConflictError(
                    f"批次编号 {batch_no!r} 已成功处理过且内容不同："
                    f"原哈希 {old_hash[:12]}… != 本次 {content_hash[:12]}…"
                )
            return BatchResult.from_json(old_result)

        try:
            result = self._run_transaction(changeset, content_hash, crash)
        except Exception:
            # 任何失败（输入校验/增量计算/落盘）都回滚，绝不留半成品
            store.conn.raw.rollback()
            raise
        return result

    # ---------------- 事务主体 ---------------- #
    def _run_transaction(self, changeset: ChangeSet,
                         content_hash: str,
                         crash: Optional[str]) -> BatchResult:
        store = self.store
        spec = self.spec
        if not changeset.changes:
            raise BatchError("批次至少包含一条修订")
        self._validate(changeset)

        # 1) 收集变更前的连接键（撤销/改 mult 需要旧值定位身份）
        initial_items: List[Tuple[str, Edge, Tuple[Any, ...]]] = []
        for ch in changeset.changes:
            if ch.op == "insert":
                key_rows = [(ch.values,)]
            else:
                phys = store.src_physical(ch.table)
                ncols = store.source_column_count(ch.table)
                row = store.conn.execute(
                    f"SELECT {', '.join(f'p{i}' for i in range(ncols))} "
                    f"FROM {phys} WHERE rid=? AND live=1",
                    (ch.rid,),
                ).fetchone()
                if row is None:
                    raise BatchError(
                        f"表 {ch.table!r} 的 rid={ch.rid} 不存在或已撤销"
                    )
                key_rows = [(tuple(row),)]
            for values, in key_rows:
                for alias, tname in spec.instances.items():
                    if tname != ch.table:
                        continue
                    for edge in self._incident_edges(alias):
                        key = self._edge_key(alias, edge, values)
                        initial_items.append((alias, edge, key))

        # 2) 在应用任何修订前，把当前约减表快照为 fd_oldred_*。
        # 脏键传播必须基于 *修订前* 的存活关系：撤销一条记录时，它原本
        # 匹配上的对侧分区仍要变脏，若在修订后的表上传播就会漏掉。
        old_red_names = self._snapshot_old_red()

        # 3) 应用源表修订（同事务；rid/row_seq 只增不复用）
        inserted_rid: Dict[int, Tuple[str, int]] = {}
        retracted_rids: List[Tuple[str, int]] = []
        updated_rids: List[Tuple[str, int]] = []
        pending_inserts: Dict[str, List[Tuple[Any, ...]]] = {}
        for pos, ch in enumerate(changeset.changes):
            if ch.op == "insert":
                pending_inserts.setdefault(ch.table, []).append(
                    (pos, ch.values, ch.mult))
            elif ch.op == "retract":
                store.mark_source_retracted(ch.table, ch.rid)
                retracted_rids.append((ch.table, ch.rid))
            else:
                phys = store.src_physical(ch.table)
                store.conn.execute(
                    f"UPDATE {phys} SET mult=? WHERE rid=?",
                    (ch.mult, ch.rid),
                )
                updated_rids.append((ch.table, ch.rid))
        for tname, items in pending_inserts.items():
            start_seq = store.next_row_seq(tname, len(items))
            for i, (pos, values, mult) in enumerate(items):
                rid = store.next_rid(tname)
                store.insert_source_record(
                    tname, rid, start_seq + i, values, mult
                )
                inserted_rid[pos] = (tname, rid)

        # 4) 脏键沿连接边在 *旧快照* 上双向传播到不动点
        self._propagate_dirty(initial_items)

        # 5) 全新约减表（只含修订后存活行）+ 边索引
        builder = _ScopedBuilder(store, spec, self)
        store.drop_reduction_tables()
        builder._create_reduction_tables()

        # 6) 仅对脏分区重做两遍半连接约减
        self._scoped_reduce(builder)
        # 传播快照用完即删（事务内，失败整体回滚）
        for name in old_red_names:
            store.conn.execute(f"DROP TABLE IF EXISTS {name}")

        # 7) 因子重建：干净分区种子化，脏分区重算
        self.loader = FactorLoader(store, spec)
        root_alias = spec.root_alias_name
        root_bnode = builder.build_subtree(root_alias, None)
        if root_bnode.kind == "empty":
            root_bnode = BEmpty(tuple(spec.output_attrs))
        output_pos = {a: i for i, a in enumerate(spec.output_attrs)}
        root_bnode = normalize(
            root_bnode, tuple(spec.output_attrs), output_pos, builder.factory
        )
        root_bnode.scope = tuple(spec.output_attrs)
        # 根可能与 memo 中的记忆化子树共享 BNode 对象；持久化按 id() 去重，
        # 直接写会让根与分区索引压扁成同一节点。给根一份结构深拷贝：
        # 普通节点取得新对象身份（新 id），种子节点保留固定旧 id，来源闭包
        # （含 PLazyRef）原样共享。
        root_bnode = _detach_root(root_bnode)
        closure = builder.make_closure()
        if root_bnode.kind != "empty":
            closure(root_bnode)
        for _a, _k, sub in builder.subtree_roots:
            if sub.kind != "empty" and sub.prov is None:
                closure(sub)

        # 7) 高水位 id + 增量持久化（旧节点行保留，种子分区写回旧 id）
        builder.node_id_start = store.factor_node_high_watermark() + 1
        builder.prov_id_start = store.prov_high_watermark() + 1
        node_hw_before = builder.node_id_start
        prov_hw_before = builder.prov_id_start
        new_root_id = builder.persist(root_bnode, incremental=True)

        # 8) 回收新版本不可达的旧因子/来源节点
        orphan_nodes, orphan_provs = self._prune_orphans(new_root_id)

        # 9) 切换根指针与版本元数据（同事务，提交前对外仍是旧版本）
        store.conn.execute(
            "DELETE FROM fd_meta WHERE key='root_node_id'"
        )
        store.conn.execute(
            "INSERT INTO fd_meta VALUES ('root_node_id', ?)",
            (str(new_root_id),)
        )
        applied_seq = store.current_applied_seq() + 1
        store.conn.execute(
            "DELETE FROM fd_meta WHERE key='current_version'"
        )
        store.conn.execute(
            "INSERT INTO fd_meta VALUES ('current_version', ?)",
            (str(applied_seq),)
        )
        total_weight = 0 if root_bnode.kind == "empty" else root_bnode.weight
        self._write_incremental_stats(builder, applied_seq, total_weight)

        result = BatchResult(
            batch_no=changeset.batch_no,
            applied_seq=applied_seq,
            version=applied_seq,
            total_weight=total_weight,
            content_hash=content_hash,
            affected_partitions={
                a: len(self.dirty_partitions[a]) for a in spec.instances
                if self.dirty_partitions[a]
            },
            rebuilt_partitions={
                a: n for a, n in self.rebuilt_count.items() if n
            },
            seeded_partitions={
                a: n for a, n in self.seeded_count.items() if n
            },
            red_rows_scanned=self.red_rows_scanned,
            red_rows_touched=self.red_rows_touched,
            factor_nodes_written=builder._next_node_id - node_hw_before,
            prov_nodes_written=builder._next_prov_id - prov_hw_before,
            prov_nodes_reused=self.loader.prov_reused,
            orphan_factor_nodes=orphan_nodes,
            orphan_prov_nodes=orphan_provs,
            inserted_rid={str(k): v for k, v in inserted_rid.items()},
            retracted_rids=retracted_rids,
            updated_rids=updated_rids,
            stats={
                "sql_statements": store.conn.statements,
                "rows_inserted": store.conn.rows_inserted,
            },
        )

        # 10) 批次日志（去重信息随事务一起落盘）
        change_rows = [
            (pos, ch.table, ch.op,
             inserted_rid.get(pos, (None, ch.rid))[1] if ch.op == "insert"
             else ch.rid,
             json.dumps(ch.payload(), ensure_ascii=False,
                        default=_json_default, sort_keys=True))
            for pos, ch in enumerate(changeset.changes)
        ]
        store.record_batch(
            changeset.batch_no, content_hash, applied_seq,
            result.to_json(), change_rows,
        )

        if crash == "before_commit":
            raise RuntimeError("模拟故障：提交前异常（事务将回滚）")
        if crash == "hard_exit_before_commit":
            # 模拟进程被强制退出：未提交事务随进程消失，重开只读到旧版本
            import os
            os._exit(7)
        store.conn.commit()
        return result

    # ---------------- 输入校验 ---------------- #
    def _validate(self, changeset: ChangeSet) -> None:
        spec = self.spec
        seen_rids: set = set()
        for pos, ch in enumerate(changeset.changes):
            if not isinstance(ch, BatchChange):
                raise BatchError(f"第 {pos} 条修订不是 BatchChange")
            if ch.op not in ("insert", "retract", "update"):
                raise BatchError(f"第 {pos} 条修订操作非法: {ch.op!r}")
            if ch.table not in spec.tables:
                raise BatchError(
                    f"第 {pos} 条修订引用未知来源表 {ch.table!r}，"
                    f"已知: {sorted(spec.tables)}"
                )
            tbl = spec.tables[ch.table]
            if ch.op == "insert":
                if not isinstance(ch.mult, int) or isinstance(ch.mult, bool) \
                        or ch.mult <= 0:
                    raise BatchError(f"第 {pos} 条插入的 mult 必须是正整数")
                if not isinstance(ch.values, tuple) or \
                        len(ch.values) != len(tbl.columns):
                    raise BatchError(
                        f"第 {pos} 条插入值列数应为 {len(tbl.columns)}"
                    )
                for col, v in zip(tbl.columns, ch.values):
                    _validate_value(col, v, pos)
            else:
                if not isinstance(ch.rid, int) or isinstance(ch.rid, bool) \
                        or ch.rid < 0:
                    raise BatchError(f"第 {pos} 条修订的 rid 必须是非负整数")
                key = (ch.table, ch.rid)
                if key in seen_rids:
                    raise BatchError(
                        f"rid={ch.rid}（表 {ch.table!r}）在同一批次被多次修订"
                    )
                seen_rids.add(key)
                if self.store.find_live_rid(ch.table, ch.rid) is None:
                    raise BatchError(
                        f"表 {ch.table!r} 的 rid={ch.rid} 不存在或已撤销，"
                        "撤销/修正只能定位存活的旧身份"
                    )
                if ch.op == "update":
                    if not isinstance(ch.mult, int) or isinstance(ch.mult, bool) \
                            or ch.mult <= 0:
                        raise BatchError(f"第 {pos} 条修正的 mult 必须是正整数")

    # ---------------- 树/键辅助 ---------------- #
    def _incident_edges(self, alias: str) -> List[Edge]:
        return [e for _v, e in self.spec.adj[alias]]

    def _edge_positions(self, alias: str, edge: Edge) -> List[int]:
        if alias == edge.alias_left:
            cols = edge.cols_left
        elif alias == edge.alias_right:
            cols = edge.cols_right
        else:
            raise BuildError("边不属于该实例")
        tname = self.spec.instances[alias]
        tbl = self.spec.tables[tname]
        return [
            next(i for i, c in enumerate(tbl.columns) if c.name == name)
            for name in cols
        ]

    def _edge_key(self, alias: str, edge: Edge,
                  values: Tuple[Any, ...]) -> Tuple[Any, ...]:
        return tuple(values[p] for p in self._edge_positions(alias, edge))

    def _snapshot_old_red(self) -> List[str]:
        """把当前各实例约减表复制为 fd_oldred_*，返回快照物理表名清单。"""
        names: List[str] = []
        self._old_red_phys: Dict[str, str] = {}
        for alias in self.spec.instances:
            src_phys = self.store.red_physical(alias)
            snap = f"fd_oldred_{alias}"
            self.store.conn.execute(f"DROP TABLE IF EXISTS {snap}")
            self.store.conn.execute(
                f"CREATE TEMP TABLE {snap} AS SELECT * FROM {src_phys}"
            )
            names.append(snap)
            self._old_red_phys[alias] = snap
        return names

    def _old_red(self, alias: str) -> str:
        return self._old_red_phys[alias]

    # ---------------- 脏键传播 ---------------- #
    def _propagate_dirty(
        self, initial: List[Tuple[str, Edge, Tuple[Any, ...]]]
    ) -> None:
        """把变更记录的连接键沿树边双向传播到不动点。

        每条边为两端各维护一个脏键集合 dirty_edge[(edge_id, alias)]。
        只有严格等值匹配（SQL = 语义，NULL 不匹配）才把键扩散到对侧；
        一个在对侧无任何匹配的孤立键只污染本侧，不会牵动干净分区。
        实例在某条边上的键变脏后，它在 *其它* 边上的键也要联动变脏。
        """
        spec = self.spec
        edge_id = {e: i for i, e in enumerate(spec.edges)}
        dirty_edge: Dict[Tuple[int, str], set] = {}
        for e in spec.edges:
            dirty_edge[(edge_id[e], e.alias_left)] = set()
            dirty_edge[(edge_id[e], e.alias_right)] = set()
        seen_side: set = set()
        stack = list(initial)
        while stack:
            alias, edge, key = stack.pop()
            eid = edge_id[edge]
            token = (eid, alias, key)
            if token in seen_side:
                continue
            seen_side.add(token)
            dirty_edge[(eid, alias)].add(key)
            other = edge.alias_right if alias == edge.alias_left \
                else edge.alias_left
            red_b = self._old_red(other)
            bp = self._edge_positions(other, edge)
            # (a) 仅当键非 NULL 且对侧有严格等值匹配时，才扩散到对侧
            if all(v is not None for v in key):
                cond = " AND ".join(f"o.p{b} = ?" for b in bp)
                cols = ", ".join(f"o.p{b}" for b in bp)
                rows = self.store.conn.execute(
                    f"SELECT DISTINCT {cols} FROM {red_b} AS o "
                    f"WHERE {cond}", key,
                ).fetchall()
                self.red_rows_scanned += len(rows)
                for nkey in rows:
                    stack.append((other, edge, tuple(nkey)))
            # (b) 本实例该键行在其它边上的键联动变脏
            red_a = self._old_red(alias)
            ap = self._edge_positions(alias, edge)
            my_cond = " AND ".join(f"t.p{p} IS ?" for p in ap)
            for e2 in self._incident_edges(alias):
                if e2 is edge:
                    continue
                p2 = self._edge_positions(alias, e2)
                cols2 = ", ".join(f"t.p{p}" for p in p2)
                rows = self.store.conn.execute(
                    f"SELECT DISTINCT {cols2} FROM {red_a} AS t "
                    f"WHERE {my_cond}", key,
                ).fetchall()
                self.red_rows_scanned += len(rows)
                for k2 in rows:
                    stack.append((alias, e2, tuple(k2)))
        # 汇总为每个实例的父边分区脏集合（子树按父键分区，根无父边）
        self.dirty_edge = dirty_edge
        for alias in spec.instances:
            pe = spec.parent_edge.get(alias)
            if pe is not None:
                self.dirty_partitions[alias] = set(
                    dirty_edge[(edge_id[pe], alias)]
                )

    # ---------------- 分区约减 ---------------- #
    def _scoped_reduce(self, builder: _ScopedBuilder) -> None:
        spec = self.spec
        # 与全量构建相同的两遍方向，但 DELETE 只作用于脏键分区
        for u in spec.postorder:
            for v in spec.children[u]:
                self._scoped_semijoin(
                    u, v, spec.child_edge[(u, v)], builder
                )
        for u in reversed(spec.postorder):
            for v in spec.children[u]:
                self._scoped_semijoin(
                    v, u, spec.child_edge[(u, v)], builder
                )

    def _scoped_semijoin(self, target: str, other: str,
                         edge: Edge, builder: _ScopedBuilder) -> None:
        spec = self.spec
        dirty_keys = self._dirty_keys_for_edge(target, edge)
        if not dirty_keys:
            return
        rt = self.store.red_physical(target)
        ro = self.store.red_physical(other)
        tp = self._edge_positions(target, edge)
        op = self._edge_positions(other, edge)
        # 先统计受影响分区的行数（核验扫描范围）
        dirty_disj = " OR ".join(
            "(" + " AND ".join(f"t.p{p} IS ?" for p in tp) + ")"
            for _k in dirty_keys
        )
        params: List[Any] = [v for k in dirty_keys for v in k]
        n = self.store.conn.execute(
            f"SELECT COUNT(*) FROM {rt} AS t WHERE {dirty_disj}", params
        ).fetchone()[0]
        self.red_rows_scanned += n
        match_cond = " AND ".join(f"o.p{o} = t.p{p}"
                                  for p, o in zip(tp, op))
        cur = self.store.conn.execute(
            f"DELETE FROM {rt} AS t WHERE ({dirty_disj}) AND NOT EXISTS "
            f"(SELECT 1 FROM {ro} AS o WHERE {match_cond})",
            params,
        )
        self.red_rows_touched += cur.rowcount if cur.rowcount != -1 else 0

    def _dirty_keys_for_edge(self, alias: str,
                             edge: Edge) -> List[Tuple[Any, ...]]:
        """该实例在指定边上的脏键集合（来自按边维护的传播结果）。"""
        eid = self._edge_id[edge]
        return list(self.dirty_edge.get((eid, alias), set()))

    # ---------------- 孤儿回收 ---------------- #
    def _prune_orphans(self, new_root_id: int) -> Tuple[int, int]:
        """回收新版本根与 *新分区索引* 都不可达的旧节点。

        必须在 fd_subtree_factor 已按新版本重写之后调用：干净分区的种子
        因子虽可能不被新根直接可达，仍被索引保留，供下一批次继续种子化，
        不能误删。
        """
        conn = self.store.conn
        reachable_nodes: set = {new_root_id}
        for (nid,) in conn.execute(
            "SELECT node_id FROM fd_subtree_factor"
        ).fetchall():
            reachable_nodes.add(nid)
        stack = list(reachable_nodes)
        while stack:
            nid = stack.pop()
            for (cid,) in conn.execute(
                "SELECT child_id FROM fd_group_entry WHERE node_id=?", (nid,)
            ).fetchall():
                if cid not in reachable_nodes:
                    reachable_nodes.add(cid)
                    stack.append(cid)
            for (cid,) in conn.execute(
                "SELECT child_id FROM fd_product_child WHERE node_id=?", (nid,)
            ).fetchall():
                if cid not in reachable_nodes:
                    reachable_nodes.add(cid)
                    stack.append(cid)
        all_nodes = {r[0] for r in conn.execute(
            "SELECT node_id FROM fd_node").fetchall()}
        orphan_n = all_nodes - reachable_nodes
        if orphan_n:
            conn.executemany(
                "DELETE FROM fd_group_entry WHERE node_id=?",
                [(n,) for n in orphan_n])
            conn.executemany(
                "DELETE FROM fd_product_child WHERE node_id=?",
                [(n,) for n in orphan_n])
            conn.executemany(
                "DELETE FROM fd_node WHERE node_id=?",
                [(n,) for n in orphan_n])

        reachable_provs: set = set()
        pstack = [r[0] for r in conn.execute(
            f"SELECT DISTINCT prov_id FROM fd_node WHERE prov_id IS NOT NULL "
            f"AND node_id IN ({','.join('?' * len(reachable_nodes))})",
            tuple(reachable_nodes),
        ).fetchall()]
        while pstack:
            pid = pstack.pop()
            if pid in reachable_provs:
                continue
            reachable_provs.add(pid)
            for (c,) in conn.execute(
                "SELECT child_prov FROM fd_prov_comp WHERE prov_id=?", (pid,)
            ).fetchall():
                pstack.append(c)
        all_provs = {r[0] for r in conn.execute(
            "SELECT prov_id FROM fd_prov_node").fetchall()}
        orphan_p = all_provs - reachable_provs
        if orphan_p:
            conn.executemany(
                "DELETE FROM fd_prov_src WHERE prov_id=?",
                [(p,) for p in orphan_p])
            conn.executemany(
                "DELETE FROM fd_prov_comp WHERE prov_id=?",
                [(p,) for p in orphan_p])
            conn.executemany(
                "DELETE FROM fd_prov_node WHERE prov_id=?",
                [(p,) for p in orphan_p])
        return len(orphan_n), len(orphan_p)

    def _write_incremental_stats(self, builder: _ScopedBuilder,
                                 version: int, total: int) -> None:
        s = self.store
        s.write_build_stat("current_version", version)
        s.write_build_stat("logical_total_weight", total)
        s.write_build_stat("last_batch_affected_partitions",
                           sum(len(v) for v in self.dirty_partitions.values()))
        s.write_build_stat("last_batch_rebuilt_partitions",
                           sum(self.rebuilt_count.values()))
        s.write_build_stat("last_batch_seeded_partitions",
                           sum(self.seeded_count.values()))
        s.write_build_stat("last_batch_red_rows_scanned",
                           self.red_rows_scanned)
        s.write_build_stat("last_batch_red_rows_touched",
                           self.red_rows_touched)


def _detach_root(root: BNode) -> BNode:
    """结构深拷贝根因子，切断与记忆化子树的对象身份共享。

    * 种子化节点（persisted_id 非 None）整体保留：旧 id 与其整棵子树
      （含 PLazyRef 来源闭包）继续复用，不进入拷贝；
    * 其余节点生成新对象，使持久化的 id() 去重把它们当作独立节点，
      来源闭包 prov 原样共享（prov DAG 本就按结构驻留，共享是安全的）。
    """
    memo: Dict[int, BNode] = {}

    def copy(node: BNode) -> BNode:
        if node.persisted_id is not None:
            return node  # 种子分区：固定旧 id，整棵子树原样复用
        cached = memo.get(id(node))
        if cached is not None:
            return cached
        if node.kind == "empty":
            new: BNode = BEmpty(node.scope)
        elif node.kind == "leaf":
            new = BLeaf(node.weight, node.prov)
        elif node.kind == "group":
            new = BGroup(
                node.attr,
                [(v, copy(c)) for v, c in node.entries],
            )
            new.scope = node.scope
        elif node.kind == "product":
            new = BProduct([copy(c) for c in node.children])
            new.scope = node.scope
        elif node.kind == "union":
            new = BUnion.__new__(BUnion)
            BNode.__init__(
                new, weight=node.weight, scope=node.scope,
                kind="union", prov=node.prov,
            )
            new.children = [copy(c) for c in node.children]
        else:
            raise BuildError(f"未知因子类型 {node.kind}")
        new.prov = node.prov
        memo[id(node)] = new
        return new

    return copy(root)


def _validate_value(col: "_RuntimeColumn", v: Any, pos: int) -> None:
    if v is None:
        if not col.nullable:
            raise BatchError(f"第 {pos} 条插入：列 {col.name!r} 不可为 NULL")
        return
    if col.dtype == "int64":
        if isinstance(v, bool) or not isinstance(v, int):
            raise BatchError(f"第 {pos} 条插入：列 {col.name!r} 需要 int64")
        if not INT64_MIN <= v <= INT64_MAX:
            raise BatchError(f"第 {pos} 条插入：列 {col.name!r} 超出 int64 范围")
    elif col.dtype == "text":
        if not isinstance(v, str):
            raise BatchError(f"第 {pos} 条插入：列 {col.name!r} 需要 text")
    else:
        if not isinstance(v, bool):
            raise BatchError(f"第 {pos} 条插入：列 {col.name!r} 需要 bool")


def apply_batch(store: DeliveryStore, changeset: ChangeSet,
                crash: Optional[str] = None) -> BatchResult:
    """在交付文件上原子地应用一个批次更新，返回可核验的处理结果。"""
    return IncrementalEngine(store).apply(changeset, crash=crash)
