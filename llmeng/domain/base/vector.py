from abc import ABC
from typing import Any, Callable, Generic, Type, TypeVar
import uuid

from loguru import logger
from pydantic import UUID4, BaseModel, Field
from qdrant_client.http import exceptions
from qdrant_client.http.models import Distance, PointStruct, VectorParams
import numpy as np

from llmeng.app.networks.embeddings import EmbeddingModelSingleton
from llmeng.domain.exceptions import ImproperlyConfigured
from llmeng.domain.types import DataCategory
from llmeng.infra.qdrant import connection

T = TypeVar("T", bound="VectorBaseDocument")


class VectorBaseDocument(BaseModel, Generic[T], ABC):
    id: UUID4 = Field(default_factory=uuid.uuid4)

    @classmethod
    def _bulk_insert(cls: Type[T], documents: list["VectorBaseDocument"]) -> None:
        points = [doc.to_point() for doc in documents]
        connection.upsert(collection_name=cls.get_collection_name(), points=points)

    @classmethod
    def _create_collection(
        cls, collection_name: str, use_vector_index: bool = True
    ) -> bool:
        if use_vector_index is True:
            vectors_config = VectorParams(
                size=EmbeddingModelSingleton().embedding_size, distance=Distance.COSINE
            )
        else:
            vectors_config = {}

        return connection.create_collection(
            collection_name=collection_name, vectors_config=vectors_config
        )

    @classmethod
    def _group_by(
        cls: Type[T], documents: list[T], selector: Callable[[T], Any]
    ) -> dict[Any, list[T]]:
        grouped = {}
        for doc in documents:
            key = selector(doc)

            if key not in grouped:
                grouped[key] = []
            grouped[key].append(doc)

        return grouped

    @classmethod
    def bulk_insert(cls: Type[T], documents: list["VectorBaseDocument"]) -> bool:
        try:
            cls._bulk_insert(documents)
        except exceptions.UnexpectedResponse:
            logger.info(
                f"Collection '{cls.get_collection_name()}' does not exist. Trying to create the collection and reinsert the documents."
            )

            cls.create_collection()

            try:
                cls._bulk_insert(documents)
            except exceptions.UnexpectedResponse:
                logger.error(
                    f"Failed to insert documents in '{cls.get_collection_name()}'."
                )

                return False

        return True

    @classmethod
    def create_collection(cls: Type[T]) -> bool:
        collection_name = cls.get_collection_name()
        use_vector_index = cls.get_use_vector_index()

        return cls._create_collection(
            collection_name=collection_name, use_vector_index=use_vector_index
        )

    @classmethod
    def get_category(cls: Type[T]) -> DataCategory:
        if not hasattr(cls, "Config") or not hasattr(cls.Config, "category"):  # type: ignore
            raise ImproperlyConfigured(
                "The class should define a Config class with"
                "the 'category' property that reflects the collection's data category."
            )

        return cls.Config.category  # type: ignore

    @classmethod
    def get_collection_name(cls: Type[T]) -> str:
        config = getattr(cls, "Config", None)
        if config is None:
            raise ImproperlyConfigured("Class should define a Config class")
        name = getattr(config, "name", None)
        if name is None:
            raise ImproperlyConfigured(
                "The class should define a Config class with"
                "the 'name' property that reflects the collection's name."
            )

        return name

    @classmethod
    def group_by_class(
        cls: Type["VectorBaseDocument"], documents: list["VectorBaseDocument"]
    ) -> dict["VectorBaseDocument", list["VectorBaseDocument"]]:
        return cls._group_by(documents, selector=lambda doc: doc.__class__)

    def to_point(self: T, **kwargs) -> PointStruct:
        exclude_unset = kwargs.pop("exclude_unset", False)
        by_alias = kwargs.pop("by_alias", True)
        payload = self.model_dump(
            exclude_unset=exclude_unset, by_alias=by_alias, **kwargs
        )
        _id = str(payload.pop("id"))
        vector = payload.pop("embedding", {})
        if vector and isinstance(vector, np.ndarray):
            vector = vector.tolist()

        return PointStruct(id=_id, vector=vector, payload=payload)

    @classmethod
    def get_use_vector_index(cls: Type[T]) -> bool:
        default = True
        config = getattr(cls, "Config", None)
        if config is None:
            return default
        return getattr(config, "use_vector_index", default)
