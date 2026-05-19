"""Datatrove examples: Hugging Face download/filter + Gemma tokenizer token count + dedup.

실행 전:
  uv sync
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from datatrove.executor.local import LocalPipelineExecutor
from datatrove.pipeline.base import PipelineStep
from datatrove.pipeline.dedup.minhash import MinhashDedupCluster, MinhashDedupSignature
from datatrove.pipeline.readers import HuggingFaceDatasetReader
from datatrove.pipeline.tokens import TokensCounter
from datatrove.pipeline.writers.jsonl import JsonlWriter


@dataclass
class MinTokensFilter(PipelineStep):
    """최소 토큰 수 미만 문서를 제거하는 필터 스텝."""

    min_tokens: int = 50

    def run(self, data, rank: int = 0, world_size: int = 1):
        for doc in data:
            token_count = doc.metadata.get("token_count", 0)
            if token_count >= self.min_tokens:
                yield doc


def build_token_count_pipeline(
    output_dir: str = "./outputs/filtered",
    cache_dir: str = "./hf_cache",
    dataset_name: str = "allenai/c4",
    split: str = "train[:1000]",
    text_key: str = "text",
):
    """HF 데이터셋 다운로드 -> Gemma4 토크나이저 토큰 수 계산 -> 필터링 -> JSONL 저장."""

    pipeline = [
        HuggingFaceDatasetReader(
            dataset=dataset_name,
            split=split,
            text_key=text_key,
            dataset_options={"cache_dir": cache_dir},
        ),
        TokensCounter(  # doc.metadata["token_count"] 저장
            tokenizer_name_or_path="google/gemma-4-9b",
            text_key=text_key,
            output_field="token_count",
        ),
        MinTokensFilter(min_tokens=50),
        JsonlWriter(output_folder=output_dir),
    ]
    return pipeline


def run_token_count_pipeline() -> None:
    executor = LocalPipelineExecutor(
        pipeline=build_token_count_pipeline(),
        tasks=4,
        logging_dir="./logs/token_count",
    )
    executor.run()


def build_dedup_pipeline(
    signature_dir: str = "./outputs/signatures",
    deduped_dir: str = "./outputs/deduped",
    cache_dir: str = "./hf_cache",
    dataset_name: str = "allenai/c4",
    split: str = "train[:1000]",
    text_key: str = "text",
):
    """HF 데이터셋에서 MinHash 기반 near-duplicate 제거 파이프라인."""

    # 1) 시그니처 생성
    signature_pipeline = [
        HuggingFaceDatasetReader(
            dataset=dataset_name,
            split=split,
            text_key=text_key,
            dataset_options={"cache_dir": cache_dir},
        ),
        MinhashDedupSignature(output_folder=signature_dir, text_key=text_key),
    ]

    # 2) 클러스터링 + 중복 제거 결과 저장
    dedup_pipeline = [
        HuggingFaceDatasetReader(
            dataset=dataset_name,
            split=split,
            text_key=text_key,
            dataset_options={"cache_dir": cache_dir},
        ),
        MinhashDedupCluster(
            input_folder=signature_dir,
            output_folder=deduped_dir,
        ),
        JsonlWriter(output_folder=deduped_dir),
    ]

    return signature_pipeline, dedup_pipeline


def run_dedup_pipeline() -> None:
    signature_pipeline, dedup_pipeline = build_dedup_pipeline()

    LocalPipelineExecutor(
        pipeline=signature_pipeline,
        tasks=4,
        logging_dir="./logs/dedup_signature",
    ).run()

    LocalPipelineExecutor(
        pipeline=dedup_pipeline,
        tasks=4,
        logging_dir="./logs/dedup_cluster",
    ).run()


if __name__ == "__main__":
    # 필요한 예시만 선택해서 실행하세요.
    run_token_count_pipeline()
    # run_dedup_pipeline()
