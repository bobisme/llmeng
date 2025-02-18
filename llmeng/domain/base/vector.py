from abc import ABC
from typing import Generic, Type, TypeVar
import uuid

from pydantic import UUID4, BaseModel, Field

from llmeng.domain.exceptions import ImproperlyConfigured
from llmeng.domain.types import DataCategory

T = TypeVar("T", bound="VectorBaseDocument")


class VectorBaseDocument(BaseModel, Generic[T], ABC):
    id: UUID4 = Field(default_factory=uuid.uuid4)

    @classmethod
    def get_category(cls: Type[T]) -> DataCategory:
        if not hasattr(cls, "Config") or not hasattr(cls.Config, "category"):  # type: ignore
            raise ImproperlyConfigured(
                "The class should define a Config class with"
                "the 'category' property that reflects the collection's data category."
            )

        return cls.Config.category  # type: ignore
