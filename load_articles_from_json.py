import asyncio
import json
import os
import re

from datasets import Dataset, DatasetDict
from litellm import Message, acompletion
from litellm.types.utils import ModelResponse
from rich.console import Console
from rich.progress import track

from llmeng.settings import settings

SENTENCE_PATTERN = re.compile(r"(?>!\w\.\w.)(?<![A-Za-z]\.)(?<=\.|\?|\!)\s")

console = Console()


def load_articles_from_json(path: str) -> Dataset:
    with open(path, "r") as file:
        data = json.load(file)

    return Dataset.from_dict(
        dict(
            id=[item["id"] for item in data],
            content=[item["content"] for item in data],
            platform=[item["platform"] for item in data],
            author_id=[item["author_id"] for item in data],
            author_full_name=[item["author_full_name"] for item in data],
            link=[item["link"] for item in data],
        )
    )


def clean_text(text: str) -> str:
    text = re.sub(r"[^\w\s.,!?']", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def extract_substring(
    dataset: Dataset, min_len: int = 1_000, max_len: int = 2_000
) -> list[str]:
    extracts = []

    for article in dataset["content"]:
        cleaned_article = clean_text(article)
        sentences = SENTENCE_PATTERN.split(cleaned_article)

        curr_chunk = ""
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            if len(curr_chunk) + len(sentence) <= max_len:
                curr_chunk += sentence + " "
            else:
                if len(curr_chunk) >= min_len:
                    extracts.append(curr_chunk.strip())
                curr_chunk = sentence + " "

        if len(curr_chunk) >= min_len:
            extracts.append(curr_chunk.strip())

    return extracts


class InstructionAnswerSet:
    def __init__(self, pairs: list[tuple[str, str]]) -> None:
        self.pairs = pairs

    @classmethod
    def from_json(cls, json_str: str) -> "InstructionAnswerSet":
        data = json.loads(json_str)
        pairs = [
            (pair["instruction"], pair["answer"])
            for pair in data["instruction_answer_pairs"]
        ]
        return cls(pairs)

    def __iter__(self):
        return iter(self.pairs)


async def generate_instruction_answer_pairs(extract: str) -> list[tuple[str, str]]:
    prompt = f"""\
Based on the following extract, generate five instruction-answer pairs. Each
instruction must ask to write about a specific topic contained in the context.
each answer must provide a relevant paragraph based on the information found in
the context. Only use concepts from the context to generate the insructions.
Instructions must never explicitly mention a context, a system, a course, or an
extract. Instructions must be self-contained and general. Answers must imitate
the writing style of the context. Example instruction: Explain the concept of
an LLM Twin. Example answer: An LLM Twin is essentially an AI character that
mimics your writing style, personality, and voice. It's designed to write just
like you by incorporating these elements into a language model. The idea is to
create a digital replica of your writing habits using advanced AI techniques.
Provide your response in JSON format with the following structure:
{{
    "instruction_answer_pairs": [
        {{"instruction": "...", "answer": "..."}},
        ...
    ]
}}
Extract:
{extract}
"""
    completion = await acompletion(
        model=settings.DATASET_GENERATION_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant who generates "
                    "instruction-answer pairs based on the given context. "
                    "Provide your response in JSON format."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        response_format={"type": "json_object"},
        max_tokens=1_200,
        temperature=0.7,
        stream=False,
    )
    assert isinstance(completion, ModelResponse), "Unexpected response"
    assert len(completion.choices) > 0, "No choices given"
    message = getattr(completion.choices[0], "message")
    assert isinstance(message, Message), "not a Message"
    if not message.content:
        raise ValueError("Did not get expected content.")
    result = InstructionAnswerSet.from_json(message.content)
    return result.pairs


async def create_instruction_dataset(dataset: Dataset, num_workers: int = 4) -> Dataset:
    extracts = extract_substring(dataset)
    instruction_answer_pairs: list[tuple[str, str]] = []

    semaphore = asyncio.Semaphore(num_workers)

    async def wrapped_generate(extract: str) -> list[tuple[str, str]]:
        async with semaphore:
            return await generate_instruction_answer_pairs(extract)

    tasks = [wrapped_generate(extract) for extract in extracts]

    for task in track(asyncio.as_completed(tasks), total=len(tasks)):
        instruction_answer_pairs.extend(await task)

    instructions, answers = zip(*instruction_answer_pairs)
    return Dataset.from_dict(
        {"instruction": list(instructions), "output": list(answers)}
    )


def main(regen: bool = False) -> DatasetDict:
    DATASET_PATH = "instruction_dataset.json"
    if not os.path.exists(DATASET_PATH) or regen:
        # Load
        raw_dataset = load_articles_from_json("./cleaned_documents.json")
        console.print("Raw dataset:")
        console.print(raw_dataset.to_pandas())

        # Create instruction set
        instruction_dataset = asyncio.run(create_instruction_dataset(raw_dataset))
        console.print("Instruction dataset:")
        console.print(instruction_dataset.to_pandas())

        console.print(f"Saving to {DATASET_PATH}")
        instruction_dataset.to_json(DATASET_PATH)
    else:
        instruction_dataset = Dataset.from_json(DATASET_PATH)
        assert isinstance(instruction_dataset, Dataset)

    # Train/Test split and export
    filtered_dataset = instruction_dataset.train_test_split(test_size=0.1)
    filtered_dataset.push_to_hub("bobnull/llmtwin")

    return filtered_dataset


if __name__ == "__main__":
    main()
