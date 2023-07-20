from __future__ import annotations

import os
import sqlite3
from threading import local
from types import TracebackType
from typing import Any
from typing import cast
from typing import Sequence
from uuid import UUID

import chromadb.db.base as base
import pypika
from chromadb.config import Settings
from chromadb.config import System
from chromadb.db.impl.sqlite_pool import Connection
from chromadb.db.impl.sqlite_pool import LockPool
from chromadb.db.impl.sqlite_pool import PerThreadPool
from chromadb.db.impl.sqlite_pool import Pool
from chromadb.db.migrations import MigratableDB
from chromadb.db.migrations import Migration
from chromadb.db.mixins.embeddings_queue import SqlEmbeddingsQueue
from chromadb.db.mixins.sysdb import SqlSysDB
from importlib_resources import files
from importlib_resources.abc import Traversable
from overrides import override
from typing_extensions import Literal


class TxWrapper(base.TxWrapper):
    _conn: Connection
    _pool: Pool

    def __init__(self, conn_pool: Pool, stack: local):
        self._tx_stack = stack
        self._conn = conn_pool.connect()
        self._pool = conn_pool

    @override
    def __enter__(self) -> base.Cursor:
        if len(self._tx_stack.stack) == 0:
            self._conn.execute("BEGIN;")
        self._tx_stack.stack.append(self)
        return self._conn.cursor()  # type: ignore

    @override
    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> Literal[False]:
        self._tx_stack.stack.pop()
        if len(self._tx_stack.stack) == 0:
            if exc_type is None:
                self._conn.commit()
            else:
                self._conn.rollback()
        self._pool.return_to_pool(self._conn)
        return False


class SqliteDB(MigratableDB, SqlEmbeddingsQueue, SqlSysDB):
    _conn_pool: Pool
    _settings: Settings
    _migration_imports: Sequence[Traversable]
    _db_file: str
    _tx_stack: local
    _is_persistent: bool

    def __init__(self, system: System):
        self._settings = system.settings
        self._migration_imports = [
            files("chromadb.migrations.embeddings_queue"),
            files("chromadb.migrations.sysdb"),
            files("chromadb.migrations.metadb"),
        ]
        self._is_persistent = self._settings.require("is_persistent")
        if not self._is_persistent:
            # In order to allow sqlite to be shared between multiple threads, we need to use a
            # URI connection string with shared cache.
            # See https://www.sqlite.org/sharedcache.html
            # https://stackoverflow.com/questions/3315046/sharing-a-memory-database-between-different-threads-in-python-using-sqlite3-pa
            self._db_file = "file::memory:?cache=shared"
            self._conn_pool = LockPool(self._db_file, is_uri=True)
        else:
            self._db_file = (
                self._settings.require("persist_directory") + "/chroma.sqlite3"
            )
            if not os.path.exists(self._db_file):
                os.makedirs(os.path.dirname(self._db_file), exist_ok=True)
            self._conn_pool = PerThreadPool(self._db_file)
        self._tx_stack = local()
        super().__init__(system)

    @override
    def start(self) -> None:
        super().start()
        with self.tx() as cur:
            cur.execute("PRAGMA foreign_keys = ON")
            cur.execute("PRAGMA case_sensitive_like = ON")
        self.initialize_migrations()

    @override
    def stop(self) -> None:
        super().stop()
        self._conn_pool.close()

    @staticmethod
    @override
    def querybuilder() -> type[pypika.Query]:
        return pypika.Query  # type: ignore

    @staticmethod
    @override
    def parameter_format() -> str:
        return "?"

    @staticmethod
    @override
    def migration_scope() -> str:
        return "sqlite"

    @override
    def migration_dirs(self) -> Sequence[Traversable]:
        return self._migration_imports

    @override
    def tx(self) -> TxWrapper:
        if not hasattr(self._tx_stack, "stack"):
            self._tx_stack.stack = []
        return TxWrapper(self._conn_pool, stack=self._tx_stack)

    @override
    def reset_state(self) -> None:
        if not self._settings.require("allow_reset"):
            raise ValueError(
                "Resetting the database is not allowed. Set `allow_reset` to true in the config in tests or other non-production environments where reset should be permitted.",
            )
        with self.tx() as cur:
            # Drop all tables
            cur.execute(
                """
                SELECT name FROM sqlite_master
                WHERE type='table'
                """,
            )
            for row in cur.fetchall():
                cur.execute(f"DROP TABLE IF EXISTS {row[0]}")
        self._conn_pool.close()
        if self._is_persistent:
            os.remove(self._db_file)
        self.start()
        super().reset_state()

    @override
    def setup_migrations(self) -> None:
        with self.tx() as cur:
            cur.execute(
                """
                 CREATE TABLE IF NOT EXISTS migrations (
                     dir TEXT NOT NULL,
                     version INTEGER NOT NULL,
                     filename TEXT NOT NULL,
                     sql TEXT NOT NULL,
                     hash TEXT NOT NULL,
                     PRIMARY KEY (dir, version)
                 )
                 """,
            )

    @override
    def migrations_initialized(self) -> bool:
        with self.tx() as cur:
            cur.execute(
                """SELECT count(*) FROM sqlite_master
                   WHERE type='table' AND name='migrations'""",
            )

            if cur.fetchone()[0] == 0:
                return False
            else:
                return True

    @override
    def db_migrations(self, dir: Traversable) -> Sequence[Migration]:
        with self.tx() as cur:
            cur.execute(
                """
                SELECT dir, version, filename, sql, hash
                FROM migrations
                WHERE dir = ?
                ORDER BY version ASC
                """,
                (dir.name,),
            )

            migrations = []
            for row in cur.fetchall():
                found_dir = cast(str, row[0])
                found_version = cast(int, row[1])
                found_filename = cast(str, row[2])
                found_sql = cast(str, row[3])
                found_hash = cast(str, row[4])
                migrations.append(
                    Migration(
                        dir=found_dir,
                        version=found_version,
                        filename=found_filename,
                        sql=found_sql,
                        hash=found_hash,
                        scope=self.migration_scope(),
                    ),
                )
            return migrations

    @override
    def apply_migration(self, cur: base.Cursor, migration: Migration) -> None:
        cur.executescript(migration["sql"])
        cur.execute(
            """
            INSERT INTO migrations (dir, version, filename, sql, hash)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                migration["dir"],
                migration["version"],
                migration["filename"],
                migration["sql"],
                migration["hash"],
            ),
        )

    @staticmethod
    @override
    def uuid_from_db(value: Any | None) -> UUID | None:
        return UUID(value) if value is not None else None

    @staticmethod
    @override
    def uuid_to_db(uuid: UUID | None) -> Any | None:
        return str(uuid) if uuid is not None else None

    @staticmethod
    @override
    def unique_constraint_error() -> type[BaseException]:
        return sqlite3.IntegrityError
