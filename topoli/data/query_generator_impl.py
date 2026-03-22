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
    queries_per_passage: int = 1,
) -> list[QueryBatch]:
    """Split passages into batches with pre-built prompts.

    When queries_per_passage > 1, each passage gets multiple prompts
    using different templates for diverse query generation.
    """
    expanded_passages: list[PassageRecord] = []
    expanded_prompts: list[str] = []
    for p in passages:
        for template_idx in range(queries_per_passage):
            expanded_passages.append(p)
            expanded_prompts.append(build_prompt(p.text, template_idx=template_idx))

    return [
        QueryBatch(
            passages=expanded_passages[i : i + batch_size],
            prompts=expanded_prompts[i : i + batch_size],
        )
        for i in range(0, len(expanded_passages), batch_size)
    ]


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
    batch_size: int = 512
    min_query_tokens: int = 5
    max_query_tokens: int = 30
    queries_per_passage: int = 2

    def generate(
        self,
        passages: list[PassageRecord],
    ) -> list[dict[str, str]]:
        """Generate queries for all passages."""
        batches = batch_passages(
            passages,
            self.batch_size,
            self.queries_per_passage,
        )
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


def build_vllm_generate_fn(
    model_name: str = "Qwen/Qwen3.5-9B",
    max_new_tokens: int = 64,
    temperature: float = 0.7,
    gpu_memory_utilization: float = 0.90,
) -> Callable[[list[str]], list[str]]:
    """Build a generate function using vLLM for high-throughput batch inference.

    vLLM uses continuous batching, PagedAttention, and CUDA graphs for
    3-5x faster throughput than naive HuggingFace generate().
    """
    from vllm import LLM, SamplingParams

    logger.info(
        "Loading %s with vLLM (gpu_mem=%.0f%%)...",
        model_name,
        gpu_memory_utilization * 100,
    )
    llm = LLM(
        model=model_name,
        dtype="bfloat16",
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=1024,
        trust_remote_code=True,
    )
    sampling_params = SamplingParams(
        temperature=temperature,
        max_tokens=max_new_tokens,
        skip_special_tokens=True,
    )
    logger.info("vLLM engine ready: %s", model_name)

    def generate(prompts: list[str]) -> list[str]:
        outputs = llm.generate(prompts, sampling_params)
        return [out.outputs[0].text.strip() for out in outputs]

    return generate


def build_hf_generate_fn(
    model_name: str = "Qwen/Qwen3.5-9B",
    max_new_tokens: int = 64,
    temperature: float = 0.7,
) -> Callable[[list[str]], list[str]]:
    """Build a generate function using HuggingFace (fallback if vLLM unavailable)."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    logger.info("Loading model %s (HF fallback)...", model_name)
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
