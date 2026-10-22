import sqlite3

from config import DB_PATH


class Connector:
    """Kết nối SQLite ở chế độ chỉ đọc (mode=ro)."""

    def __init__(self, db_path=DB_PATH):
        self.db_path = str(db_path)
        self.conn = None

    def connect(self):
        if self.conn is None:
            self.conn = sqlite3.connect(f"file:{self.db_path}?mode=ro", uri=True)
            self.conn.row_factory = sqlite3.Row
        return self.conn

    def close(self):
        if self.conn is not None:
            self.conn.close()
            self.conn = None

    def __enter__(self):
        return self.connect()

    def __exit__(self, *exc):
        self.close()
