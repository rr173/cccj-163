"""SQLite 持久层。

磁盘协议（可审查的节点表，见 README「因子节点协议」）：

fd_meta             键值元数据（协议版本、规格名、总重数校验值等）
fd_source_table     来源表登记
fd_source_column    来源表列定义（位置、名字、类型、可空、逻辑属性）
fd_instance         表实例（角色别名 -> 来源表；同一来源可承担多个角色）
fd_instance_column  实例列定义及在来源表中的物理位置
fd_edge             连接边（边号、端点、左右位置、键列、逻辑属性）
fd_output_attr      输出列的逻辑属性与定义顺序
fd_node             因子节点：kind ∈ group/product/leaf/empty/union，缓存重数、作用域、
                    来源闭包 prov_id（empty 节点为 NULL）
fd_group_entry      group 节点的排序值域（每行一个取值及其子节点）
fd_product_child    product/union 节点的分支（按输出顺序 / 袋并次序）
fd_prov_node        来源表达节点：kind ∈ src/prod/alt，缓存来源组合数（weight）
fd_prov_src         src 节点的来源行清单（实例别名、输入行序号、该行 multiplicity）
fd_prov_comp        prod/alt 节点的子来源节点（op 标明相乘还是袋并）
fd_build_stat       构建统计（节点数、各表行数、SQL 语句数等）

源表物理名 src_<i>，约减表物理名 red_<alias 编号>，均只保存 *存活* 元组，
绝不保存完整宽连接结果。两张表都带 row_seq：来源行按装载顺序编号（从 0 起），
同一张表被多个实例引用时各自保留独立身份；值完全相同的行序号不同、仍是不同来源行。
"""

from __future__ import annotations

import sqlite3
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .spec import Spec

PROTOCOL_VERSION = "2.0"


class ProtocolError(RuntimeError):
    """磁盘文件不符合因子节点协议。"""


class CountingConn:
    """带语句计数的 SQLite 连接包装。"""

    def __init__(self, conn: sqlite3.Connection):
        self._conn = conn
        self.statements = 0
        self.rows_inserted = 0

    def execute(self, sql: str, params: Sequence[Any] = ()) -> sqlite3.Cursor:
        self.statements += 1
        return self._conn.execute(sql, params)

    def executemany(self, sql: str, seq: Sequence[Sequence[Any]]) -> sqlite3.Cursor:
        self.statements += 1
        n = len(seq)
        self.rows_inserted += n
        return self._conn.executemany(sql, seq)

    def executescript(self, sql: str) -> sqlite3.Cursor:
        # DDL 脚本不计入数据访问语句数
        return self._conn.executescript(sql)

    def commit(self) -> None:
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    @property
    def raw(self) -> sqlite3.Connection:
        return self._conn


SCHEMA_SQL = """
CREATE TABLE fd_meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE fd_source_table (
    table_name TEXT PRIMARY KEY
);
CREATE TABLE fd_source_column (
    table_name TEXT NOT NULL,
    position  INTEGER NOT NULL,
    col_name  TEXT NOT NULL,
    dtype     TEXT NOT NULL,
    nullable  INTEGER NOT NULL,
    attr      TEXT NOT NULL,
    PRIMARY KEY (table_name, position)
);
CREATE TABLE fd_instance (
    alias      TEXT PRIMARY KEY,
    table_name TEXT NOT NULL
);
CREATE TABLE fd_instance_column (
    alias           TEXT NOT NULL,
    position        INTEGER NOT NULL,
    col_name        TEXT NOT NULL,
    attr            TEXT NOT NULL,
    dtype           TEXT NOT NULL,
    nullable        INTEGER NOT NULL,
    source_position INTEGER NOT NULL,
    PRIMARY KEY (alias, position)
);
CREATE TABLE fd_edge (
    edge_id        INTEGER NOT NULL,
    endpoint_alias TEXT NOT NULL,
    side           TEXT NOT NULL CHECK (side IN ('L', 'R')),
    position       INTEGER NOT NULL,
    col_name       TEXT NOT NULL,
    attr           TEXT NOT NULL,
    PRIMARY KEY (edge_id, endpoint_alias, position)
);
CREATE TABLE fd_output_attr (
    position INTEGER PRIMARY KEY,
    attr     TEXT NOT NULL,
    dtype    TEXT NOT NULL,
    nullable INTEGER NOT NULL
);
CREATE TABLE fd_node (
    node_id INTEGER PRIMARY KEY,
    kind    TEXT NOT NULL CHECK (kind IN ('group', 'product', 'leaf', 'empty', 'union')),
    weight  TEXT NOT NULL,
    scope   TEXT NOT NULL,
    prov_id INTEGER
);
CREATE TABLE fd_group_entry (
    node_id    INTEGER NOT NULL,
    ord        INTEGER NOT NULL,
    attr       TEXT NOT NULL,
    is_null    INTEGER NOT NULL,
    v_int      INTEGER,
    v_text     TEXT,
    v_bool     INTEGER,
    child_id   INTEGER NOT NULL,
    PRIMARY KEY (node_id, ord)
);
CREATE INDEX idx_group_entry_node ON fd_group_entry(node_id, ord);
CREATE TABLE fd_product_child (
    node_id  INTEGER NOT NULL,
    position INTEGER NOT NULL,
    child_id INTEGER NOT NULL,
    PRIMARY KEY (node_id, position)
);
-- 来源追溯（lineage / provenance）：紧凑来源表达 DAG。
-- src：某个表实例上的一组来源行（行序号 + 该行 multiplicity，不逐份展开）；
-- prod：各分支共同贡献，来源组合数相乘（同一连接组合内的各来源）；
-- alt：袋并，来源组合数相加（投影后多个连接分支得到同一输出元组，
--       不同连接键分组的来源在此保持互不交叉）。
CREATE TABLE fd_prov_node (
    prov_id INTEGER PRIMARY KEY,
    kind    TEXT NOT NULL CHECK (kind IN ('src', 'prod', 'alt')),
    weight  TEXT NOT NULL
);
CREATE TABLE fd_prov_src (
    prov_id  INTEGER NOT NULL,
    alias    TEXT NOT NULL,
    row_seq  INTEGER NOT NULL,
    mult     INTEGER NOT NULL,
    position INTEGER NOT NULL,
    PRIMARY KEY (prov_id, position)
);
CREATE INDEX idx_prov_src_lookup ON fd_prov_src(alias, row_seq);
CREATE TABLE fd_prov_comp (
    prov_id     INTEGER NOT NULL,
    position    INTEGER NOT NULL,
    child_prov  INTEGER NOT NULL,
    PRIMARY KEY (prov_id, position)
);
CREATE INDEX idx_prov_comp_child ON fd_prov_comp(child_prov);
CREATE TABLE fd_build_stat (
    metric TEXT PRIMARY KEY,
    value  TEXT NOT NULL
);
-- 批次更新日志：按 batch_no 去重；content_hash 是批次全部内容的稳定哈希。
-- status='committed' 的批次重放时：同哈希返回原结果，不同哈希明确拒绝。
CREATE TABLE fd_batch_log (
    batch_no     TEXT PRIMARY KEY,
    content_hash TEXT NOT NULL,
    status       TEXT NOT NULL CHECK (status IN ('committed')),
    applied_seq  INTEGER NOT NULL,
    result_json  TEXT NOT NULL,
    created_at   TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE TABLE fd_batch_change (
    batch_no     TEXT NOT NULL,
    position     INTEGER NOT NULL,
    table_name   TEXT NOT NULL,
    op           TEXT NOT NULL CHECK (op IN ('insert', 'retract', 'update')),
    rid          INTEGER,
    payload_json TEXT NOT NULL,
    PRIMARY KEY (batch_no, position)
);
-- 构建期子树因子索引：每个 (实例, 父键分区) 记忆化因子对应一个持久化
-- 因子节点（规范化前子树的根），增量更新时干净分区凭它种子化复用，
-- 无需重算该分区，也不枚举任何组合。
CREATE TABLE fd_subtree_factor (
    alias        TEXT NOT NULL,
    parent_key   TEXT NOT NULL,
    node_id      INTEGER NOT NULL,
    PRIMARY KEY (alias, parent_key)
);
-- 各来源表的 row_seq 高水位：新记录序号只增不减，撤销/同值插入都不回收。
CREATE TABLE fd_seq_watermark (
    table_name TEXT PRIMARY KEY,
    next_seq   INTEGER NOT NULL
);
"""


class DeliveryStore:
    """因子交付文件。新建时装载规格与源表，重开时仅读协议元数据。"""

    def __init__(self, path: str, conn: CountingConn, owns: bool = True):
        self.path = path
        self.conn = conn
        self._owns = owns
        self._output_attrs: Optional[List[Tuple[str, str, bool]]] = None
        self._attr_meta: Optional[Dict[str, Tuple[str, bool]]] = None
        self._instances: Optional[Dict[str, str]] = None
        self._instance_dtypes: Optional[Dict[str, List[Tuple[str, str, bool]]]] = None

    # ------------------------------------------------------------------ #
    @classmethod
    def create(cls, path: str, spec: Spec) -> "DeliveryStore":
        raw = sqlite3.connect(path)
        raw.execute("PRAGMA journal_mode=DELETE")
        conn = CountingConn(raw)
        conn.executescript(SCHEMA_SQL)
        store = cls(path, conn)
        spec.validate()
        store._write_spec(spec)
        store.conn.commit()
        return store

    @classmethod
    def open(cls, path: str) -> "DeliveryStore":
        raw = sqlite3.connect(path)
        conn = CountingConn(raw)
        store = cls(path, conn)
        store._verify_protocol()
        store._load_meta()
        return store

    def close(self) -> None:
        if self._owns:
            self.conn.commit()
            self.conn.close()

    def __enter__(self) -> "DeliveryStore":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    # ------------------------------------------------------------------ #
    def _verify_protocol(self) -> None:
        cur = self.conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='fd_node'"
        )
        if cur.fetchone() is None:
            raise ProtocolError(f"{self.path} 不是因子交付文件（缺少 fd_node 表）")
        cur = self.conn.execute(
            "SELECT value FROM fd_meta WHERE key='protocol_version'"
        )
        row = cur.fetchone()
        if row is None or row[0] != PROTOCOL_VERSION:
            raise ProtocolError("协议版本缺失或不受支持")
        cur = self.conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='fd_prov_node'"
        )
        if cur.fetchone() is None:
            raise ProtocolError("交付文件缺少来源追溯表 fd_prov_node")

    def _load_meta(self) -> None:
        cur = self.conn.execute(
            "SELECT attr, dtype, nullable FROM fd_output_attr ORDER BY position"
        )
        self._output_attrs = [(a, t, bool(n)) for a, t, n in cur.fetchall()]
        self._attr_meta = {a: (t, n) for a, t, n in self._output_attrs}
        # 条件里允许出现的属性只需是输出属性；条件只能筛选输出列。
        # 来源追溯需要实例身份与来源表映射，重开后从协议表恢复。
        cur = self.conn.execute("SELECT alias, table_name FROM fd_instance")
        self._instances = {alias: tname for alias, tname in cur.fetchall()}
        self._instance_dtypes = {}
        for alias in self._instances:
            cur = self.conn.execute(
                "SELECT col_name, dtype, nullable FROM fd_instance_column "
                "WHERE alias=? ORDER BY position",
                (alias,),
            )
            self._instance_dtypes[alias] = [
                (name, dtype, bool(nullable)) for name, dtype, nullable in cur.fetchall()
            ]

    @property
    def instances(self) -> Dict[str, str]:
        """实例别名 -> 来源表名（同一来源表可被多个实例分别引用）。"""
        assert self._instances is not None
        return dict(self._instances)

    def instance_columns(self, alias: str) -> List[Tuple[str, str, bool]]:
        """实例各列的 (列名, 声明类型, 是否可空)，按来源表装载位置排列。"""
        assert self._instance_dtypes is not None
        if alias not in self._instance_dtypes:
            raise ProtocolError(f"未知表实例 {alias!r}")
        return list(self._instance_dtypes[alias])

    def source_row_count(self, alias: str) -> int:
        """实例来源表中 *存活* 来源行的数量（撤销行不计；序号空间仍为
        0..高水位-1，已撤销序号在审计查询中返回 None）。"""
        assert self._instances is not None
        if alias not in self._instances:
            raise ProtocolError(f"未知表实例 {alias!r}")
        phys = self.src_physical(self._instances[alias])
        return self.conn.execute(
            f"SELECT COUNT(*) FROM {phys} WHERE live=1"
        ).fetchone()[0]

    def row_seq_high_watermark(self, alias: str) -> int:
        """实例来源表的 row_seq 高水位（含已撤销记录，只增不减）。

        合法序号空间为 0..高水位-1；其中部分序号可能对应已撤销记录。
        """
        assert self._instances is not None
        if alias not in self._instances:
            raise ProtocolError(f"未知表实例 {alias!r}")
        phys = self.src_physical(self._instances[alias])
        row = self.conn.execute(
            f"SELECT COALESCE(MAX(row_seq) + 1, 0) FROM {phys}"
        ).fetchone()
        return row[0]

    def fetch_source_row(self, alias: str, row_seq: int) -> Optional[Dict[str, Any]]:
        """按装载序号取一条 *存活* 来源行（含 mult 与各列值，bool 已还原）。

        序号越界或该序号对应记录已被撤销时返回 None，便于审计者直接核对
        fd_prov_src 里登记的来源（新版本的来源表达不会再引用撤销行）。
        """
        assert self._instances is not None and self._instance_dtypes is not None
        if alias not in self._instances:
            raise ProtocolError(f"未知表实例 {alias!r}")
        if not isinstance(row_seq, int) or isinstance(row_seq, bool) or row_seq < 0:
            raise ProtocolError("row_seq 必须是非负整数")
        cols = self._instance_dtypes[alias]
        phys = self.src_physical(self._instances[alias])
        select = ", ".join(["row_seq", "mult"] + [f"p{i}" for i in range(len(cols))])
        row = self.conn.execute(
            f"SELECT {select} FROM {phys} WHERE row_seq=? AND live=1",
            (row_seq,),
        ).fetchone()
        if row is None:
            return None
        values = {}
        for i, (name, dtype, _nullable) in enumerate(cols):
            raw = row[2 + i]
            values[name] = bool(raw) if dtype == "bool" and raw is not None else raw
        return {"row_seq": row[0], "mult": row[1], "values": values}

    @property
    def output_attrs(self) -> List[Tuple[str, str, bool]]:
        assert self._output_attrs is not None
        return list(self._output_attrs)

    def attr_meta(self, attr: str) -> Tuple[str, bool]:
        assert self._attr_meta is not None
        if attr not in self._attr_meta:
            raise ProtocolError(f"{attr!r} 不是输出列，条件只能作用于输出列")
        return self._attr_meta[attr]

    # ------------------------------------------------------------------ #
    def _write_spec(self, spec: Spec) -> None:
        c = self.conn
        c.execute("INSERT INTO fd_meta VALUES ('protocol_version', ?)",
                  (PROTOCOL_VERSION,))
        c.execute("INSERT INTO fd_meta VALUES ('spec_name', ?)", (spec.name,))
        c.execute("INSERT INTO fd_meta VALUES ('current_version', ?)", ("0",))

        # 来源表与列
        for ti, (tname, tbl) in enumerate(spec.tables.items()):
            c.execute("INSERT INTO fd_source_table VALUES (?)", (tname,))
            for pos, col in enumerate(tbl.columns):
                c.execute(
                    "INSERT INTO fd_source_column VALUES (?, ?, ?, ?, ?, ?)",
                    (tname, pos, col.name, col.dtype, 1 if col.nullable else 0,
                     col.attr),
                )
            phys = self.src_physical(tname)
            col_defs = ", ".join(
                f"p{pos} {_sqlite_type(col.dtype)}" for pos, col in enumerate(tbl.columns)
            )
            # row_seq：来源行装载序号（从 0 起），来源追溯以此命名来源行。
            # rid：稳定记录身份，主键；live=0 是已撤销记录（保留身份历史，
            # 不再参与任何版本的查询）。初始装载 rid 与 row_seq 相同。
            c.execute(
                f"CREATE TABLE {phys} (rid INTEGER NOT NULL, "
                "row_seq INTEGER NOT NULL, live INTEGER NOT NULL DEFAULT 1, "
                f"{col_defs}, mult INTEGER NOT NULL, PRIMARY KEY (rid))"
            )
            c.execute(
                f"CREATE UNIQUE INDEX idx_{phys}_rowseq ON {phys}(row_seq)"
            )
            params = [(i, i, 1, *row, mult) for i, (row, mult) in
                      enumerate(zip(tbl.rows, tbl.multiplicities))]
            c.executemany(
                f"INSERT INTO {phys} VALUES "
                f"({','.join('?' * (len(tbl.columns) + 4))})",
                params,
            )
            c.execute(
                "INSERT INTO fd_seq_watermark VALUES (?, ?)",
                (tname, len(tbl.rows)),
            )

        # 实例与实例列
        for alias, tname in spec.instances.items():
            c.execute("INSERT INTO fd_instance VALUES (?, ?)", (alias, tname))
            tbl = spec.tables[tname]
            for pos, col in enumerate(tbl.columns):
                c.execute(
                    "INSERT INTO fd_instance_column VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (alias, pos, col.name, col.attr, col.dtype,
                     1 if col.nullable else 0, pos),
                )

        # 连接边
        for eid, e in enumerate(spec.edges):
            ta = spec.tables[spec.instances[e.alias_left]]
            tb = spec.tables[spec.instances[e.alias_right]]
            for pos, (cl, cr) in enumerate(zip(e.cols_left, e.cols_right)):
                attr = ta.col(cl).attr
                c.execute(
                    "INSERT INTO fd_edge VALUES (?, ?, ?, ?, ?, ?)",
                    (eid, e.alias_left, "L", pos, cl, attr),
                )
                c.execute(
                    "INSERT INTO fd_edge VALUES (?, ?, ?, ?, ?, ?)",
                    (eid, e.alias_right, "R", pos, cr, attr),
                )

        # 输出列
        for pos, attr in enumerate(spec.output_attrs):
            c.execute(
                "INSERT INTO fd_output_attr VALUES (?, ?, ?, ?)",
                (pos, attr, spec.attr_type[attr],
                 1 if spec.attr_nullable[attr] else 0),
            )

    # ------------------------------------------------------------------ #
    @staticmethod
    def src_physical(table_name: str) -> str:
        # 表名经登记后用稳定别名，避免标识符注入问题
        return "src_" + _ident(table_name)

    @staticmethod
    def red_physical(alias: str) -> str:
        return "red_" + _ident(alias)

    def create_reduction_table(self, alias: str, ncols: int) -> str:
        phys = self.red_physical(alias)
        col_defs = ", ".join(f"p{i}" for i in range(ncols))
        # 保留来源行装载序号 row_seq：约减只删行、不重排，序号始终指向 src 表
        self.conn.execute(
            f"CREATE TABLE {phys} (row_seq INTEGER NOT NULL, {col_defs}, "
            "mult INTEGER NOT NULL, PRIMARY KEY (row_seq))"
        )
        return phys

    def drop_reduction_tables(self) -> None:
        """删除全部约减表及其索引（增量更新开始时清理上一版本中间结果）。"""
        for alias in self.instances:
            phys = self.red_physical(alias)
            for (idx_name,) in self.conn.execute(
                "SELECT name FROM sqlite_master WHERE type='index' "
                "AND tbl_name=?", (phys,)
            ).fetchall():
                self.conn.execute(f"DROP INDEX IF EXISTS {idx_name}")
            self.conn.execute(f"DROP TABLE IF EXISTS {phys}")

    def drop_factor_tables(self) -> None:
        self.conn.execute("DELETE FROM fd_node")
        self.conn.execute("DELETE FROM fd_group_entry")
        self.conn.execute("DELETE FROM fd_product_child")
        self.conn.execute("DELETE FROM fd_prov_node")
        self.conn.execute("DELETE FROM fd_prov_src")
        self.conn.execute("DELETE FROM fd_prov_comp")
        self.conn.execute("DELETE FROM fd_subtree_factor")
        self.conn.execute("DELETE FROM fd_build_stat")

    def write_build_stat(self, metric: str, value: Any) -> None:
        self.conn.execute(
            "INSERT INTO fd_build_stat VALUES (?, ?) "
            "ON CONFLICT(metric) DO UPDATE SET value=excluded.value",
            (metric, str(value)),
        )

    def read_build_stats(self) -> Dict[str, str]:
        cur = self.conn.execute("SELECT metric, value FROM fd_build_stat")
        return dict(cur.fetchall())

    # ------------------------------------------------------------------ #
    # 批次更新支持
    # ------------------------------------------------------------------ #
    def lookup_batch(self, batch_no: str) -> Optional[tuple]:
        """已提交批次 -> (content_hash, applied_seq, result_json)；未见返回 None。"""
        row = self.conn.execute(
            "SELECT content_hash, applied_seq, result_json FROM fd_batch_log "
            "WHERE batch_no=?",
            (batch_no,),
        ).fetchone()
        return None if row is None else tuple(row)

    def record_batch(self, batch_no: str, content_hash: str,
                     applied_seq: int, result_json: str,
                     changes: List[Tuple[int, str, str, Any, str]]) -> None:
        """在当前事务内登记批次日志与逐变更明细。

        changes: [(position, table_name, op, rid, payload_json), ...]
        """
        self.conn.execute(
            "INSERT INTO fd_batch_log (batch_no, content_hash, status, "
            "applied_seq, result_json) VALUES (?, ?, 'committed', ?, ?)",
            (batch_no, content_hash, applied_seq, result_json),
        )
        self.conn.executemany(
            "INSERT INTO fd_batch_change (batch_no, position, table_name, op, "
            "rid, payload_json) VALUES (?, ?, ?, ?, ?, ?)",
            [(batch_no, pos, tname, op, rid, payload)
             for pos, tname, op, rid, payload in changes],
        )

    def list_batches(self) -> List[Dict[str, Any]]:
        """按应用顺序列出全部已提交批次（重开后仍可审计批次去重信息）。"""
        cur = self.conn.execute(
            "SELECT batch_no, content_hash, applied_seq, result_json, created_at "
            "FROM fd_batch_log ORDER BY applied_seq"
        )
        return [
            {"batch_no": b, "content_hash": h, "applied_seq": s,
             "result_json": rj, "created_at": ts}
            for b, h, s, rj, ts in cur.fetchall()
        ]

    def next_row_seq(self, table_name: str, n: int) -> int:
        """在当前事务内领取 n 个连续 row_seq，返回起始序号并推进高水位。"""
        row = self.conn.execute(
            "SELECT next_seq FROM fd_seq_watermark WHERE table_name=?",
            (table_name,),
        ).fetchone()
        if row is None:
            raise ProtocolError(f"未知来源表 {table_name!r}")
        start = row[0]
        self.conn.execute(
            "UPDATE fd_seq_watermark SET next_seq=? WHERE table_name=?",
            (start + n, table_name),
        )
        return start

    def insert_source_record(self, table_name: str, rid: int, row_seq: int,
                             values: Sequence[Any], mult: int) -> None:
        ncols = self.source_column_count(table_name)
        self.conn.execute(
            f"INSERT INTO {self.src_physical(table_name)} "
            f"(rid, row_seq, live, {', '.join(f'p{i}' for i in range(ncols))}, "
            "mult) VALUES (" + ",".join("?" * (ncols + 4)) + ")",
            (rid, row_seq, 1, *values, mult),
        )

    def mark_source_retracted(self, table_name: str, rid: int) -> None:
        self.conn.execute(
            f"UPDATE {self.src_physical(table_name)} SET live=0 WHERE rid=?",
            (rid,),
        )

    def source_column_count(self, table_name: str) -> int:
        row = self.conn.execute(
            "SELECT COUNT(*) FROM fd_source_column WHERE table_name=?",
            (table_name,),
        ).fetchone()
        return row[0]

    def source_table_names(self) -> List[str]:
        return [r[0] for r in self.conn.execute(
            "SELECT table_name FROM fd_source_table ORDER BY table_name"
        ).fetchall()]

    def next_rid(self, table_name: str) -> int:
        """新记录身份：现有 rid 最大值 + 1（撤销记录的 rid 永不复用）。"""
        row = self.conn.execute(
            f"SELECT COALESCE(MAX(rid), -1) + 1 FROM "
            f"{self.src_physical(table_name)}"
        ).fetchone()
        return row[0]

    def find_live_rid(self, table_name: str, rid: int) -> Optional[int]:
        row = self.conn.execute(
            f"SELECT rid FROM {self.src_physical(table_name)} "
            "WHERE rid=? AND live=1",
            (rid,),
        ).fetchone()
        return None if row is None else row[0]

    def clear_subtree_factors(self) -> None:
        self.conn.execute("DELETE FROM fd_subtree_factor")

    def write_subtree_factor(self, alias: str,
                             parent_key: Tuple[Any, ...], node_id: int) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO fd_subtree_factor VALUES (?, ?, ?)",
            (alias, _key_text(parent_key), node_id),
        )

    def read_subtree_factor_id(self, alias: str,
                               parent_key: Tuple[Any, ...]) -> Optional[int]:
        row = self.conn.execute(
            "SELECT node_id FROM fd_subtree_factor WHERE alias=? AND parent_key=?",
            (alias, _key_text(parent_key)),
        ).fetchone()
        return None if row is None else row[0]

    def current_applied_seq(self) -> int:
        row = self.conn.execute(
            "SELECT COALESCE(MAX(applied_seq), 0) FROM fd_batch_log"
        ).fetchone()
        return row[0]

    def factor_node_high_watermark(self) -> int:
        row = self.conn.execute(
            "SELECT COALESCE(MAX(node_id), 0) FROM fd_node"
        ).fetchone()
        return row[0]

    def prov_high_watermark(self) -> int:
        row = self.conn.execute(
            "SELECT COALESCE(MAX(prov_id), 0) FROM fd_prov_node"
        ).fetchone()
        return row[0]

    def file_size_bytes(self) -> int:
        self.conn.commit()
        cur = self.conn.execute("PRAGMA page_count")
        pages = cur.fetchone()[0]
        cur = self.conn.execute("PRAGMA page_size")
        page_size = cur.fetchone()[0]
        return pages * page_size


def _sqlite_type(dtype: str) -> str:
    return {"int64": "INTEGER", "text": "TEXT", "bool": "INTEGER"}[dtype]


def _ident(name: str) -> str:
    out = []
    for ch in name:
        if ch.isalnum() or ch == "_":
            out.append(ch)
        else:
            out.append(f"_{ord(ch):x}_")
    return "".join(out)


def _key_text(key: Sequence[Any]) -> str:
    """父键分区的稳定文本编码（类型感知，bool 与 int 不混淆）。"""
    parts = []
    for v in key:
        if v is None:
            parts.append("n")
        if isinstance(v, bool):
            parts.append(f"b:{int(v)}")
        elif isinstance(v, int):
            parts.append(f"i:{v}")
        else:
            parts.append("s:" + str(v).replace("\\", "\\\\"))
    return "|".join(parts)
