"""SQLite 持久层。

磁盘协议（可审查的节点表，见 README「因子节点协议」）：

fd_meta             键值元数据（协议版本、规格名、总重数校验值等）
fd_source_table     来源表登记
fd_source_column    来源表列定义（位置、名字、类型、可空、逻辑属性）
fd_instance         表实例（角色别名 -> 来源表；同一来源可承担多个角色）
fd_instance_column  实例列定义及在来源表中的物理位置
fd_edge             连接边（边号、端点、左右位置、键列、逻辑属性）
fd_output_attr      输出列的逻辑属性与定义顺序
fd_node             因子节点：kind ∈ group/product/leaf/empty，缓存重数、作用域
fd_group_entry      group 节点的排序值域（每行一个取值及其子节点）
fd_product_child    product 节点的不重叠分支（按输出顺序排列）
fd_build_stat       构建统计（节点数、各表行数、SQL 语句数等）

源表物理名 src_<i>，约减表物理名 red_<alias 编号>，均只保存 *存活* 元组，
绝不保存完整宽连接结果。
"""

from __future__ import annotations

import sqlite3
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .spec import Spec

PROTOCOL_VERSION = "1.0"


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
    scope   TEXT NOT NULL
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
CREATE TABLE fd_build_stat (
    metric TEXT PRIMARY KEY,
    value  TEXT NOT NULL
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

    def _load_meta(self) -> None:
        cur = self.conn.execute(
            "SELECT attr, dtype, nullable FROM fd_output_attr ORDER BY position"
        )
        self._output_attrs = [(a, t, bool(n)) for a, t, n in cur.fetchall()]
        self._attr_meta = {a: (t, n) for a, t, n in self._output_attrs}
        # 条件里允许出现的属性只需是输出属性；条件只能筛选输出列。

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
            c.execute(f"CREATE TABLE {phys} ({col_defs}, mult INTEGER NOT NULL)")
            params = [(*row, mult) for row, mult in
                      zip(tbl.rows, tbl.multiplicities)]
            c.executemany(
                f"INSERT INTO {phys} VALUES ({','.join('?' * (len(tbl.columns) + 1))})",
                params,
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
        self.conn.execute(
            f"CREATE TABLE {phys} ({col_defs}, mult INTEGER NOT NULL)"
        )
        return phys

    def drop_factor_tables(self) -> None:
        self.conn.execute("DELETE FROM fd_node")
        self.conn.execute("DELETE FROM fd_group_entry")
        self.conn.execute("DELETE FROM fd_product_child")
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
