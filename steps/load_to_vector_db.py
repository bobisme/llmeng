from typing import Annotated
from loguru import logger
from zenml import step

from llmeng import utils
from llmeng.domain.base.vector import VectorBaseDocument


@step
def load_to_vector_db(documents: Annotated[list, "documents"]) -> bool:
    logger.info(f"Loading {len(documents)} documents into the vector database")

    grouped_documents = VectorBaseDocument.group_by_class(documents)
    for doc_class, documents in grouped_documents.items():
        logger.info(f"Loading documents into {doc_class.get_collection_name()}")
        for docs_batch in utils.batch(documents, size=4):
            try:
                doc_class.bulk_insert(docs_batch)
            except Exception:
                return False
    return True
