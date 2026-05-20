from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from datatrove.executor.local import LocalPipelineExecutor
from datatrove.pipeline.base import PipelineStep
from datatrove.pipeline.dedup.minhash import (
    MinhashConfig,
    MinhashDedupBuckets,
    MinhashDedupCluster,
    MinhashDedupFilter,
    MinhashDedupSignature,
)
from datatrove.pipeline.readers import HuggingFaceDatasetReader, JsonlReader
from datatrove.pipeline.writers.jsonl import JsonlWriter
from transformers import AutoTokenizer

DEFAULT_DATASETS = [
    "blythet/deepseek-v4-pro-math-cot-1k",
    "nvidia/Nemotron-SFT-Math-v3",
    "angrygiraffe/claude-opus-4.6-4.7-reasoning-8.7k",
    "Jackrong/GLM-5.1-Reasoning-1M-Cleaned",
    "Jackrong/Kimi-K2.5-Reasoning-1M-Cleaned",
    "Jongsim/claude-opus-4.6-reasoning-12k-en-filtered-v2",
]

PROMPT_KEYS = ("prompt", "question", "instruction", "input", "query", "problem", "user", "text")


@dataclass
class PromptExtractor(PipelineStep):
    source_dataset: str

    def run(self, data, rank: int = 0, world_size: int = 1):
        for doc in data:
            row = doc.metadata or {}
            doc.text = pick_prompt_text(row, doc.text)
            doc.metadata["source_dataset"] = self.source_dataset
            yield doc


@dataclass
class TokenCountCollector(PipelineStep):
    tokenizer_name: str

    def __post_init__(self):
        self._tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name, use_fast=True)

    def run(self, data, rank: int = 0, world_size: int = 1):
        for doc in data:
            ids = self._tokenizer(doc.text or "", add_special_tokens=False, truncation=False)["input_ids"]
            doc.metadata["token_count"] = len(ids)
            yield doc


def normalize_ws(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "")).strip()


def flatten_messages(messages: Any) -> str:
    if not isinstance(messages, list):
        return ""
    chunks: list[str] = []
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        if str(msg.get("role", "")).lower() not in {"user", "human"}:
            continue
        content = msg.get("content", "")
        if isinstance(content, str):
            chunks.append(content)
        elif isinstance(content, list):
            nested = [c.get("text", "") for c in content if isinstance(c, dict)]
            if nested:
                chunks.append("\n".join(nested))
    return normalize_ws("\n".join(chunks))


def pick_prompt_text(row: dict[str, Any], fallback_text: str) -> str:
    if "messages" in row:
        text = flatten_messages(row.get("messages"))
        if text:
            return text

    for k in ("conversation", "conversations", "dialog", "chat"):
        if k in row:
            text = flatten_messages(row.get(k))
            if text:
                return text

    for k in PROMPT_KEYS:
        v = row.get(k)
        if isinstance(v, str) and v.strip():
            return normalize_ws(v)
        if isinstance(v, dict):
            nested = v.get("text") or v.get("content")
            if isinstance(nested, str) and nested.strip():
                return normalize_ws(nested)

    return normalize_ws(fallback_text)


def per_dataset_pipeline(dataset: str, output_dir: Path, tokenizer: str, split: str, cache_dir: str):
    slug = dataset.replace("/", "__")
    return [
        HuggingFaceDatasetReader(dataset=dataset, split=split, text_key="text", dataset_options={"cache_dir": cache_dir}),
        PromptExtractor(source_dataset=dataset),
        TokenCountCollector(tokenizer_name=tokenizer),
        JsonlWriter(output_folder=str(output_dir / "prepared"), output_filename=f"{slug}_${{rank}}.jsonl"),
    ]


def write_token_stats(prepared_dir: Path, out_file: Path) -> None:
    token_sum: dict[str, int] = {}
    row_sum: dict[str, int] = {}
    for p in prepared_dir.glob("*.jsonl"):
        with p.open(encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                source = obj.get("metadata", {}).get("source_dataset", "unknown")
                token_count = int(obj.get("metadata", {}).get("token_count", 0))
                token_sum[source] = token_sum.get(source, 0) + token_count
                row_sum[source] = row_sum.get(source, 0) + 1

    rows = []
    for source in sorted(row_sum):
        cnt = row_sum[source]
        total = token_sum[source]
        rows.append({
            "dataset": source,
            "rows": cnt,
            "total_tokens_approx": total,
            "avg_tokens_per_row": (total / cnt) if cnt else 0,
        })

    with out_file.open("w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)


def build_dedup_pipeline(prepared_dir: Path, output_dir: Path, similarity_threshold: float):
    config = MinhashConfig()
    config.threshold = similarity_threshold

    signatures = output_dir / "signatures"
    buckets = output_dir / "buckets"
    remove_ids = output_dir / "remove_ids"

    signature_pipeline = [
        JsonlReader(data_folder=str(prepared_dir), text_key="text"),
        MinhashDedupSignature(output_folder=str(signatures), config=config),
    ]
    buckets_pipeline = [MinhashDedupBuckets(input_folder=str(signatures), output_folder=str(buckets), config=config)]
    cluster_pipeline = [MinhashDedupCluster(input_folder=str(buckets), output_folder=str(remove_ids), config=config)]
    filter_pipeline = [
        JsonlReader(data_folder=str(prepared_dir), text_key="text"),
        MinhashDedupFilter(input_folder=str(remove_ids), exclusion_writer=JsonlWriter(output_folder=str(output_dir / "removed"))),
        JsonlWriter(output_folder=str(output_dir / "deduped")),
    ]
    return signature_pipeline, buckets_pipeline, cluster_pipeline, filter_pipeline


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", required=True)
    p.add_argument("--cache-dir", default="./hf_cache")
    p.add_argument("--split", default="train")
    p.add_argument("--tokenizer", default="google/gemma-3-4b-it")
    p.add_argument("--datasets", nargs="*", default=DEFAULT_DATASETS)
    p.add_argument("--tasks", type=int, default=4)
    p.add_argument("--similarity-threshold", type=float, default=0.92)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    base = Path(args.output_dir)
    base.mkdir(parents=True, exist_ok=True)

    for ds in args.datasets:
        LocalPipelineExecutor(
            pipeline=per_dataset_pipeline(ds, base, args.tokenizer, args.split, args.cache_dir),
            tasks=args.tasks,
            logging_dir=str(base / "logs" / ds.replace("/", "__") / "prepare"),
        ).run()

    prepared_dir = base / "prepared"
    write_token_stats(prepared_dir, base / "token_stats.json")

    sig_p, buck_p, clu_p, fil_p = build_dedup_pipeline(prepared_dir, base, args.similarity_threshold)
    LocalPipelineExecutor(pipeline=sig_p, tasks=args.tasks, logging_dir=str(base / "logs" / "signature")).run()
    LocalPipelineExecutor(pipeline=buck_p, tasks=args.tasks, logging_dir=str(base / "logs" / "buckets")).run()
    LocalPipelineExecutor(pipeline=clu_p, tasks=args.tasks, logging_dir=str(base / "logs" / "cluster")).run()
    LocalPipelineExecutor(pipeline=fil_p, tasks=args.tasks, logging_dir=str(base / "logs" / "filter")).run()


if __name__ == "__main__":
    main()
