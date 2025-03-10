#!/usr/bin/env python

import comet_ml
import torch

from trl import SFTTrainer
from datasets import load_dataset, concatenate_datasets
from transformers import TrainingArguments
from unsloth import FastLanguageModel, is_bfloat16_supported

MAX_SEQ_LEN = 2048
MODEL_NAME = "meta-llama/Meta-Llama-3.1-8B"
ALPACA_TEMPLATE = """\
Below is an instruction that describes a task.
Write a response that appropriately completes the request.

### Instruction:
{}

### Response:
{}"""

model, tokenizer = FastLanguageModel.from_pretrained(
    # The base model to load - Llama 3.1 8B parameter version
    # model_name="meta-llama/Meta-Llama-3.1-8B",
    model_name=MODEL_NAME,
    # Maximum sequence length for input/output (affects context window and memory usage)
    max_seq_length=MAX_SEQ_LEN,
    # When False, loads in full precision (FP16/BF16); when True, uses 4-bit
    # quantization to reduce memory
    load_in_4bit=False,
)
# model = AutoModelForCausalLM.from_pretrained(
#     MODEL,
#     attn_implementation="flash_attention_2" if is_bfloat16_supported() else "sdpa",
#     torch_dtype=torch.bfloat16 if is_bfloat16_supported() else torch.float16,
# )

# tokenizer = AutoTokenizer.from_pretrained(MODEL)
if "Qwen2.5-7B" in MODEL_NAME:
    tokenizer.pad_token = "<|im_end|>"
    tokenizer.pad_token_id = 151645
    tokenizer.padding_side = "left"
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs=dict(use_reentrant=True)
    )
    model.max_seq_length = MAX_SEQ_LEN

EOS_TOKEN = tokenizer.eos_token

model = FastLanguageModel.get_peft_model(
    model,
    # Rank of the LoRA adapters - higher values = more capacity but increased
    # memory usage
    r=32,
    # Scaling factor for LoRA - controls how much influence the adapters have
    # (often set equal to r)
    lora_alpha=32,
    # Dropout rate for LoRA layers - 0 means no dropout, higher values like 0.1
    # add regularization
    lora_dropout=0,
    target_modules=[
        "q_proj",  # Query projection in attention mechanism
        "k_proj",  # Key projection in attention mechanism
        "v_proj",  # Value projection in attention mechanism
        "up_proj",  # Upward projection in MLP/feedforward blocks
        "down_proj",  # Downward projection in MLP/feedforward blocks
        "o_proj",  # Output projection in attention mechanism
        "gate_proj",  # Gate projection in MLP blocks (for SwiGLU activations)
    ],
)

my_dataset = load_dataset("bobnull/llmtwin", split="train")
# > Next, we need to prepare the data in the right format for fine-tuning. In
# > this example, we don’t have a lot of samples in the llmtwin dataset (3,000
# > samples). This is an issue because the model might not correctly learn the
# > chat template. To address this, we will upsample it with a high-quality
# > general-purpose dataset called FineTome. This is a filtered version of
# > arcee-ai/The-Tome using the fineweb-edu-classifier. Instead of using the
# > 100,000 samples of this dataset, we will specify we only want 10,000 in the
# > train split. We concatenate these two datasets to create our final set.
supplemental_dataset = load_dataset(
    "mlabonne/FineTome-Alpaca-100k", split="train[:10000]"
)
dataset = concatenate_datasets([my_dataset, supplemental_dataset])


def format_samples(samples):
    text = []
    for instruction, output in zip(
        samples["instruction"], samples["output"], strict=False
    ):
        message = ALPACA_TEMPLATE.format(instruction, output) + EOS_TOKEN
        text.append(message)

    return {"text": text}


dataset = dataset.map(format_samples, batched=True, remove_columns=dataset.column_names)
dataset = dataset.train_test_split(test_size=0.05)

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    # # tokenizer=tokenizer,
    dataset_text_field="text",
    max_seq_length=MAX_SEQ_LEN,
    dataset_num_proc=2,
    packing=True,
    args=TrainingArguments(
        learning_rate=3e-4,
        lr_scheduler_type="linear",
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        num_train_epochs=3,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.01,
        warmup_steps=10,
        output_dir="output",
        report_to="comet_ml",
        seed=0,
    ),
)

trainer.train()

model.save_pretrained_merged("model", tokenizer, save_method="merged_16bit")
model.push_to_hub_merged(
    "bobnull/TwinLlama-3.1-8B", tokenizer, save_method="merged_16bit"
)
