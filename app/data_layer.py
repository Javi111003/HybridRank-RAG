"""SQLite data layer for Chainlit chat persistence."""

import os
import sqlite3

import chainlit as cl
from chainlit.data.sql_alchemy import SQLAlchemyDataLayer


class SQLiteDataLayer(SQLAlchemyDataLayer):
    async def create_element(self, element):
        pass

    async def delete_element(self, element_id, thread_id=None):
        pass

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
os.makedirs(DATA_DIR, exist_ok=True)
DB_PATH = os.path.abspath(os.path.join(DATA_DIR, "chat_history.db"))

FILES_DIR = os.path.join(os.path.dirname(__file__), "..", ".files")
os.makedirs(FILES_DIR, exist_ok=True)

CONNINFO = f"sqlite+aiosqlite:///{DB_PATH}"

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS users (
    "id" TEXT PRIMARY KEY,
    "identifier" TEXT UNIQUE NOT NULL,
    "createdAt" TEXT,
    "metadata" TEXT DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS threads (
    "id" TEXT PRIMARY KEY,
    "createdAt" TEXT,
    "name" TEXT,
    "userId" TEXT,
    "userIdentifier" TEXT,
    "tags" TEXT,
    "metadata" TEXT DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS steps (
    "id" TEXT PRIMARY KEY,
    "name" TEXT,
    "type" TEXT,
    "threadId" TEXT,
    "parentId" TEXT,
    "streaming" INTEGER DEFAULT 0,
    "waitForAnswer" INTEGER DEFAULT 0,
    "isError" INTEGER DEFAULT 0,
    "metadata" TEXT DEFAULT '{}',
    "tags" TEXT,
    "input" TEXT DEFAULT '',
    "output" TEXT DEFAULT '',
    "createdAt" TEXT,
    "start" TEXT,
    "end" TEXT,
    "generation" TEXT,
    "showInput" TEXT,
    "indent" INTEGER,
    "language" TEXT,
    "command" TEXT,
    "defaultOpen" INTEGER DEFAULT 0,
    "autoCollapse" INTEGER DEFAULT 0,
    "icon" TEXT,
    "modes" TEXT
);

CREATE TABLE IF NOT EXISTS elements (
    "id" TEXT PRIMARY KEY,
    "threadId" TEXT,
    "type" TEXT,
    "url" TEXT,
    "chainlitKey" TEXT,
    "name" TEXT,
    "display" TEXT,
    "objectKey" TEXT,
    "size" TEXT,
    "page" INTEGER,
    "language" TEXT,
    "forId" TEXT,
    "mime" TEXT,
    "props" TEXT
);

CREATE TABLE IF NOT EXISTS feedbacks (
    "id" TEXT PRIMARY KEY,
    "forId" TEXT,
    "threadId" TEXT,
    "value" INTEGER,
    "comment" TEXT,
    "strategy" TEXT
);
"""


def _init_db_sync():
    conn = sqlite3.connect(DB_PATH)
    conn.executescript(SCHEMA_SQL)
    # Migration: add columns that may be missing from older schema
    for col, default in [("autoCollapse", "0"), ("icon", "NULL")]:
        try:
            conn.execute(f'ALTER TABLE steps ADD COLUMN "{col}" {"INTEGER DEFAULT " + default if default.isdigit() else "TEXT"}')
        except sqlite3.OperationalError:
            pass  # Column already exists
    conn.close()


_init_db_sync()


@cl.data_layer
def get_data_layer():
    return SQLiteDataLayer(conninfo=CONNINFO, storage_provider=None)
