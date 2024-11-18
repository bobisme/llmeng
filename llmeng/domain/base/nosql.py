import uuid
from abc import ABC
from typing import Generic, Type, TypeVar, Protocol
from typing_extensions import ClassVar
import sqlite3
import json

from loguru import logger
from pydantic import UUID4, BaseModel, Field

from llmeng.nosql import db


class DocumentSettings(Protocol):
    name: ClassVar[str]


class HasSettings(Protocol):
    Settings: ClassVar[DocumentSettings]


T = TypeVar("T", bound="NoSQLBaseDocument")
DocumentType = TypeVar("DocumentType", bound=HasSettings)


class NoSQLBaseDocument(BaseModel, Generic[T], ABC):
    id: UUID4 = Field(default_factory=uuid.uuid4)

    def __eq__(self, value: object) -> bool:
        if not isinstance(value, self.__class__):
            return False

        return self.id == value.id

    def __hash__(self) -> int:
        return hash(self.id)

    @classmethod
    def from_sqlite(cls: Type[T], data: dict) -> T:
        """Convert SQLite JSON data into a document instance."""
        if not data:
            raise ValueError("Data is empty.")

        # Extract the _id and convert the JSON data
        doc_data = json.loads(data["data"])
        doc_data["id"] = doc_data.pop("_id")

        return cls(**doc_data)

    def to_sqlite(self: T, **kwargs) -> tuple[str, str, str]:
        """Convert the document to SQLite format (id, collection, JSON data)."""
        exclude_unset = kwargs.pop("exclude_unset", False)
        by_alias = kwargs.pop("by_alias", True)

        parsed = self.model_dump(
            exclude_unset=exclude_unset, by_alias=by_alias, **kwargs
        )

        # Convert id to _id for storage
        if "id" in parsed:
            parsed["_id"] = str(parsed.pop("id"))

        # Convert UUID fields to strings
        for key, value in parsed.items():
            if isinstance(value, uuid.UUID):
                parsed[key] = str(value)

        return (str(self.id), self.get_collection_name(), json.dumps(parsed))

    # def save(self: T, **kwargs) -> T | None:
    def save(self: T, **kwargs) -> T:
        """Save the document to SQLite."""
        try:
            with db.get_connection() as conn:
                cursor = conn.cursor()
                doc_id, collection, data = self.to_sqlite(**kwargs)

                cursor.execute(
                    """
                    INSERT OR REPLACE INTO documents (_id, collection, data)
                    VALUES (?, ?, ?)
                """,
                    (doc_id, collection, data),
                )

                conn.commit()
                return self
        except sqlite3.Error as e:
            logger.exception(f"Failed to insert document: {e}")
            raise

    @classmethod
    def find(cls: Type[T], **filter_options) -> T | None:
        """Find a single document matching the filter options."""
        try:
            with db.get_connection() as conn:
                cursor = conn.cursor()

                # Convert filter options to JSON query conditions
                conditions = []
                params = [cls.get_collection_name()]

                for key, value in filter_options.items():
                    if isinstance(value, uuid.UUID):
                        value = str(value)
                    conditions.append(f"json_extract(data, '$.{key}') = ?")
                    params.append(value)

                query = """
                    SELECT data FROM documents 
                    WHERE collection = ?
                """

                if conditions:
                    query += " AND " + " AND ".join(conditions)

                cursor.execute(query, params)
                result = cursor.fetchone()

                if result:
                    return cls.from_sqlite({"data": result[0]})
                return None

        except sqlite3.Error as e:
            logger.error(f"Failed to retrieve document: {e}")
            return None

    @classmethod
    def get_or_create(cls: Type[T], **filter_options) -> T:
        """Get an existing document or create a new one."""
        try:
            # Try to find existing document
            instance = cls.find(**filter_options)
            if instance:
                return instance

            # Create new instance if not found
            new_instance = cls(**filter_options)
            new_instance = new_instance.save()
            return new_instance

        except sqlite3.Error as e:
            logger.exception(f"Failed to get or create document: {e}")
            raise

    @classmethod
    def get_collection_name(cls: Type[DocumentType]) -> str:
        """Get the collection name from Settings."""
        if not hasattr(cls, "Settings") or not hasattr(cls.Settings, "name"):
            raise RuntimeError(
                "Document should define a Settings configuration class with the name of the collection."
            )
        return cls.Settings.name

    @classmethod
    def bulk_insert(cls: Type[T], documents: list[T], **kwargs) -> bool:
        """Insert multiple documents at once."""
        try:
            with db.get_connection() as conn:
                cursor = conn.cursor()
                values = [doc.to_sqlite(**kwargs) for doc in documents]

                cursor.executemany(
                    """
                    INSERT OR REPLACE INTO documents (_id, collection, data)
                    VALUES (?, ?, ?)
                """,
                    values,
                )

                conn.commit()
                return True

        except sqlite3.Error as e:
            logger.error(f"Failed to insert documents of type {cls.__name__}: {e}")
            return False

    @classmethod
    def bulk_find(cls: Type[T], **filter_options) -> list[T]:
        """Find all documents matching the filter options."""
        try:
            with db.get_connection() as conn:
                cursor = conn.cursor()

                # Convert filter options to JSON query conditions
                conditions = []
                params = [cls.get_collection_name()]

                for key, value in filter_options.items():
                    if isinstance(value, uuid.UUID):
                        value = str(value)
                    conditions.append(f"json_extract(data, '$.{key}') = ?")
                    params.append(value)

                query = """
                    SELECT data FROM documents 
                    WHERE collection = ?
                """

                if conditions:
                    query += " AND " + " AND ".join(conditions)

                cursor.execute(query, params)
                results = cursor.fetchall()

                return [
                    document
                    for result in results
                    if (document := cls.from_sqlite({"data": result[0]})) is not None
                ]

        except sqlite3.Error as e:
            logger.error(f"Failed to retrieve documents: {e}")
            return []
