from typing import Annotated, Any
from zenml import get_step_context, step

from llmeng.app.preprocessing.dispatchers import CleaningDispatcher
from llmeng.domain.cleaned_documents import CleanedDocument


def _get_metadata(cleaned_documents: list[CleanedDocument]) -> dict:
    metadata: dict[Any, Any] = {"num_documents": len(cleaned_documents)}
    for document in cleaned_documents:
        category = document.get_category()
        if category not in metadata:
            metadata[category] = {}
        if "authors" not in metadata[category]:
            metadata[category]["authors"] = list()

        metadata[category]["num_documents"] = (
            metadata[category].get("num_documents", 0) + 1
        )
        metadata[category]["authors"].append(document.author_full_name)

    for value in metadata.values():
        if isinstance(value, dict) and "authors" in value:
            value["authors"] = list(set(value["authors"]))

    return metadata


@step
def clean_documents(
    docs: Annotated[list, "raw_documents"],
) -> Annotated[list, "clean_documents"]:
    cleaned_documents = []
    for doc in docs:
        cleaned_doc = CleaningDispatcher.dispatch(doc)
        cleaned_documents.append(cleaned_doc)
    step_context = get_step_context()
    step_context.add_output_metadata(
        output_name="clean_documents", metadata=_get_metadata(cleaned_documents)
    )
    return cleaned_documents
