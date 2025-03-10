#!/usr/bin/env python

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TextStreamer,
)

MODEL_NAME = "bobnull/TwinLlama-3.1-8B"
MAX_SEQ_LEN = 2048
ALPACA_TEMPLATE = """\
Below is an instruction that describes a task.
Write a response that appropriately completes the request.

### Instruction:
{}

### Response:
{}"""

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16)
model.to("cuda")
message = ALPACA_TEMPLATE.format(
    "Write a paragraph to introduce supervised fine-tuning", ""
)
inputs = tokenizer([message], return_tensors="pt").to("cuda")

text_streamer = TextStreamer(tokenizer)
_ = model.generate(
    **inputs,
    streamer=text_streamer,
    max_new_tokens=256,
    cache_implementation="offloaded",
)
