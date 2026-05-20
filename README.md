# Hugging Face 데이터셋 필터링 파이프라인 (Datatrove)

이 저장소는 datatrove를 이용해 아래 작업을 한 번에 수행합니다.

1. 각 데이터셋에서 **prompt/question 계열 텍스트만 추출**
2. **Gemma 토크나이저 기반 대략 토큰 수 집계**
3. 전체 데이터 병합 후 **MinHash near-duplicate 제거**

기본 대상 데이터셋:

- `blythet/deepseek-v4-pro-math-cot-1k`
- `nvidia/Nemotron-SFT-Math-v3`
- `angrygiraffe/claude-opus-4.6-4.7-reasoning-8.7k`
- `Jackrong/GLM-5.1-Reasoning-1M-Cleaned`
- `Jackrong/Kimi-K2.5-Reasoning-1M-Cleaned`
- `Jongsim/claude-opus-4.6-reasoning-12k-en-filtered-v2`

---

## 설치

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync
```

---

## 실행

기본 토크나이저는 `google/gemma-3-4b-it` 입니다.

```bash
uv run python pipeline_examples.py \
  --output-dir ./outputs/math_reasoning
```

커스텀 토크나이저/저장 경로/유사도 임계치 지정:

```bash
uv run python pipeline_examples.py \
  --output-dir /data/math_dedup \
  --tokenizer google/gemma-3-4b-it \
  --similarity-threshold 0.92 \
  --tasks 8
```

특정 데이터셋만 지정하고 싶다면:

```bash
uv run python pipeline_examples.py \
  --output-dir ./outputs/custom \
  --datasets nvidia/Nemotron-SFT-Math-v3 Jongsim/claude-opus-4.6-reasoning-12k-en-filtered-v2
```

---

## 주요 옵션

- `--output-dir` (필수): 결과 저장 경로
- `--tokenizer` (기본: `google/gemma-3-4b-it`): 토큰 카운트용 토크나이저
- `--similarity-threshold` (기본: `0.92`): MinHash 중복 판단 임계치
- `--datasets` (기본: 위 6개): 처리할 HF 데이터셋 목록
- `--split` (기본: `train`): HF split
- `--tasks` (기본: `4`): 병렬 작업 수
- `--cache-dir` (기본: `./hf_cache`): HF 캐시 경로

---

## 출력물

`--output-dir` 하위에 다음이 생성됩니다.

- `prepared/*.jsonl`:
  - prompt/question 추출 + `token_count` 메타데이터 포함 중간 결과
- `token_stats.json`:
  - 데이터셋별 행 수/총 토큰(근사)/평균 토큰
- `deduped/*.jsonl`:
  - 유사도 높은 prompt 제거 후 최종 결과
- `removed/*.jsonl`:
  - 중복으로 제거된 샘플
- `logs/*`:
  - 단계별 datatrove 실행 로그

---

## 동작 방식

- prompt 추출 시 `messages`/`conversation` 구조가 있으면 user/human 발화만 우선 사용
- 그렇지 않으면 `prompt`, `question`, `instruction`, `input` 등 우선순위 키를 사용
- 정답/추론(assistant/solution/cot) 자체를 dedup 기준으로 사용하지 않고, **질문 측 텍스트 중심**으로 dedup

---

## 주의사항

- 토큰 수는 학습 파이프라인용 **근사치**이며 모델 내부 exact counting과 100% 동일하지 않을 수 있습니다.
- 대규모 데이터셋에서는 디스크/메모리 사용량이 큽니다. `--tasks`와 split 범위를 먼저 작게 검증하세요.
