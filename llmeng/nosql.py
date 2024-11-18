from contextlib import contextmanager
import sqlite3


class DatabaseConnectionManager:
    _instance = None
    _initialized = False

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, database_path: str = "llmeng.db"):
        if not self._initialized and database_path:
            self.database_path = database_path
            self.initialize()
            DatabaseConnectionManager._initialized = True

    @contextmanager
    def get_connection(self):
        if not self._initialized:
            raise RuntimeError(
                "DatabaseConnectionManager not initialized. Call initialize() first."
            )

        conn = sqlite3.connect(self.database_path)
        # Enable JSON1 extension features
        conn.enable_load_extension(True)
        try:
            yield conn
        finally:
            conn.close()

    def initialize(self):
        """Initialize the database with required tables."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    _id TEXT PRIMARY KEY,
                    collection TEXT NOT NULL,
                    data JSON NOT NULL
                )
            """
            )
            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_collection 
                ON documents(collection)
            """
            )
            conn.commit()


# Global connection manager instance
db = DatabaseConnectionManager()
