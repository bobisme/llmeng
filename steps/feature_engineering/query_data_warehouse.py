from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Annotated, Any

from zenml import get_step_context, step
from loguru import logger

from llmeng.domain.base.nosql import NoSQLBaseDocument
from llmeng.domain.documents import (
    ArticleDocument,
    Document,
    PostDocument,
    RepositoryDocument,
    UserDocument,
)
from llmeng.utils import split_user_full_name


def _get_metadata(docs: list[Document]) -> dict:
    metadata: dict[Any, Any] = {
        "num_documents": len(docs),
    }
    for doc in docs:
        collection = doc.get_collection_name()
        if collection not in metadata:
            metadata[collection] = {}
        if "authors" not in metadata[collection]:
            metadata[collection]["authors"] = list()

        metadata[collection]["num_documents"] = (
            metadata[collection].get("num_documents", 0) + 1
        )
        metadata[collection]["authors"].append(doc.author_full_name)

    for value in metadata.values():
        if isinstance(value, dict) and "authors" in value:
            value["authors"] = list(set(value["authors"]))

    return metadata


def __fetch_articles(user_id: str) -> list[NoSQLBaseDocument]:
    return ArticleDocument.bulk_find(author_id=user_id)


def __fetch_posts(user_id: str) -> list[NoSQLBaseDocument]:
    return PostDocument.bulk_find(author_id=user_id)


def __fetch_repositories(user_id: str) -> list[NoSQLBaseDocument]:
    return RepositoryDocument.bulk_find(author_id=user_id)


def fetch_all_data(user: UserDocument) -> dict[str, list[NoSQLBaseDocument]]:
    user_id = str(user.id)
    with ThreadPoolExecutor() as executor:
        futures_to_query = {
            executor.submit(__fetch_articles, user_id): "articles",
            executor.submit(__fetch_posts, user_id): "posts",
            executor.submit(__fetch_repositories, user_id): "repositories",
        }
        results = {}
        for future in as_completed(futures_to_query):
            query_name = futures_to_query[future]
            try:
                results[query_name] = future.result()
            except Exception:
                logger.exception(f"'{query_name}' request failed")
                results[query_name] = []
    return results


@step
def query_data_warehouse(
    author_full_names: list[str],
) -> Annotated[list, "raw_documents"]:
    docs = []
    authors = []
    for author_full_name in author_full_names:
        logger.info(f"Querying data warehouse for {author_full_name}")
        first_name, last_name = split_user_full_name(author_full_name)
        logger.info(f"First name: {first_name}, Last name: {last_name}")
        user = UserDocument.get_or_create(first_name=first_name, last_name=last_name)
        authors.append(user)
        results = fetch_all_data(user)
        user_docs = [doc for query_result in results.values() for doc in query_result]
        docs.extend(user_docs)

    step_context = get_step_context()
    step_context.add_output_metadata(
        output_name="raw_documents", metadata=_get_metadata(docs)
    )
    return docs
