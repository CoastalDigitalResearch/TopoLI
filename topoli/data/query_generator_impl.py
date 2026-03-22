"""Real query generation: batched inference with open-weight LLMs.

Generates search queries from passages using Qwen3-8B (Apache-2.0)
or any HuggingFace causal LM. Designed for GPU batch inference on A100.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass

from topoli.data.query_generator import build_prompt, parse_query_response
from topoli.data.source_config import License, PassageRecord

logger = logging.getLogger(__name__)


@dataclass
class QueryBatch:
    """A batch of passages with pre-built prompts for inference."""

    passages: list[PassageRecord]
    prompts: list[str]


def batch_passages(
    passages: list[PassageRecord],
    batch_size: int = 32,
) -> list[QueryBatch]:
    """Split passages into batches with pre-built prompts."""
    batches: list[QueryBatch] = []
    for i in range(0, len(passages), batch_size):
        chunk = passages[i : i + batch_size]
        prompts = [build_prompt(p.text) for p in chunk]
        batches.append(QueryBatch(passages=chunk, prompts=prompts))
    return batches


@dataclass
class QueryGenPipeline:
    """Pipeline for generating queries from passages.

    Takes a generate_fn that maps prompts -> responses.
    For real usage, this wraps a HuggingFace model.
    For testing, pass a mock function.
    """

    generate_fn: Callable[[list[str]], list[str]]
    model_name: str
    model_license: License
    batch_size: int = 32
    min_query_tokens: int = 5
    max_query_tokens: int = 30

    def generate(
        self,
        passages: list[PassageRecord],
    ) -> list[dict[str, str]]:
        """Generate queries for all passages."""
        batches = batch_passages(passages, self.batch_size)
        results: list[dict[str, str]] = []

        for batch_idx, batch in enumerate(batches):
            responses = self.generate_fn(batch.prompts)

            for passage, response in zip(batch.passages, responses, strict=True):
                query = parse_query_response(
                    response,
                    min_tokens=self.min_query_tokens,
                    max_tokens=self.max_query_tokens,
                )
                if query is None:
                    continue

                results.append(
                    {
                        "query": query,
                        "passage_id": passage.passage_id,
                        "passage_text": passage.text,
                        "source_name": passage.source_name,
                        "source_license": passage.source_license.value,
                        "generator_model": self.model_name,
                        "generator_license": self.model_license.value,
                    }
                )

            if (batch_idx + 1) % 100 == 0:
                logger.info(
                    "Generated %d queries from %d batches",
                    len(results),
                    batch_idx + 1,
                )

        logger.info(
            "Total queries generated: %d / %d passages", len(results), len(passages)
        )
        return results


def build_hf_generate_fn(
    model_name: str = "Qwen/Qwen3-8B",
    max_new_tokens: int = 64,
    temperature: float = 0.7,
) -> Callable[[list[str]], list[str]]:
    """Build a generate function using a HuggingFace causal LM on GPU."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    logger.info("Loading model %s...", model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    logger.info("Model loaded: %s", model_name)

    def generate(prompts: list[str]) -> list[str]:
        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
            )

        responses: list[str] = []
        for i, output in enumerate(outputs):
            input_len = inputs["input_ids"][i].shape[0]
            new_tokens = output[input_len:]
            decoded = tokenizer.decode(new_tokens, skip_special_tokens=True)
            text = decoded if isinstance(decoded, str) else " ".join(decoded)
            responses.append(text.strip())

        return responses

    return generate
