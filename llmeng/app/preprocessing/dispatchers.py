from loguru import logger

from llmeng.domain.base.nosql import NoSQLBaseDocument
from llmeng.domain.base.vector import VectorBaseDocument
from llmeng.domain.types import DataCategory


from .cleaning_data_handlers import (
    CleaningDataHandler,
    ArticleCleaningHandler,
    PostCleaningHandler,
    RepositoryCleaningHandler,
)


class CleaningHandlerFactory:
    @staticmethod
    def create_handler(data_category: DataCategory) -> CleaningDataHandler:
        match data_category:
            case DataCategory.POSTS:
                return PostCleaningHandler()
            case DataCategory.ARTICLES:
                return ArticleCleaningHandler()
            case DataCategory.REPOSITORIES:
                return RepositoryCleaningHandler()
            case _:
                raise ValueError("Unsupported data type")


class CleaningDispatcher:
    factory = CleaningHandlerFactory()

    @classmethod
    def dispatch(cls, data_model: NoSQLBaseDocument) -> VectorBaseDocument:
        data_category = DataCategory(data_model.get_collection_name())
        handler = cls.factory.create_handler(data_category)
        clean_model = handler.clean(data_model)

        logger.info(
            "Document cleaned",
            data_category=data_category,
            cleanded_content_len=len(clean_model.content),
        )

        return clean_model
