from loguru import logger
from typing import Annotated
from zenml import step

from llmeng import utils
from llmeng.domain.base.vector import VectorBaseDocument


@step
def load_to_vector_db(
    documents: Annotated[list, "documents"],
) -> Annotated[bool, "successful"]:
    logger.info(f"Loading {len(documents)} documents into the vector db")

    grouped_documents = VectorBaseDocument.group_by_class(documents)
    for doc_class, docs in grouped_documents.items():
        logger.info(f"Loading documents into {doc_class.get_collection_name()}")
        for docs_batch in utils.batch(docs, size=4):
            try:
                doc_class.bulk_insert(docs_batch)
            except Exception as error:
                logger.error(
                    f"Failed to insert docs into {doc_class.get_collection_name()}: {error}",
                )
                return False
    return True
