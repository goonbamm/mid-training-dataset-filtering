# Hugging Face 데이터셋 필터링 파이프라인 (Datatrove 예시)

이 저장소는 **Hugging Face 데이터셋을 다운로드**하고,

1. **Gemma4 토크나이저로 토큰 수를 측정**한 뒤 필터링하는 예시
2. **문서 간 near-duplicate(유사 중복) 제거** 예시

를 `datatrove` 기반으로 제공합니다.

---

## 1) 설치 (uv 기반)

```bash
# uv가 없다면 설치 (macOS/Linux)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 의존성 동기화 (.venv 자동 생성)
uv sync
```

> `pyproject.toml`에 `datatrove`, `datasets`, `transformers`, `sentencepiece`가 정의되어 있습니다.

---

## 2) 파일 구조

- `pipeline_examples.py`
  - Hugging Face Reader
  - Gemma 토큰 수 카운팅 + 최소 토큰 필터
  - MinHash 기반 중복 제거 파이프라인 예시
- `pyproject.toml`
  - uv/프로젝트 의존성 관리

---

## 3) Gemma4 토크나이저로 토큰 수 측정 + 필터링

`pipeline_examples.py`의 `build_token_count_pipeline()` / `run_token_count_pipeline()`을 사용합니다.

핵심 흐름:

1. `HuggingFaceDatasetReader`로 데이터셋 로드
2. `TokensCounter(tokenizer_name_or_path="google/gemma-4-9b")`로 토큰 수 계산
3. `MinTokensFilter(min_tokens=50)`로 짧은 문서 제거
4. `JsonlWriter`로 결과 저장

실행:

```bash
uv run python pipeline_examples.py
```

기본값 기준 결과:

- 입력: `allenai/c4`, `train[:1000]`
- 출력: `./outputs/filtered`
- 로그: `./logs/token_count`

원하는 데이터셋으로 바꾸려면 `build_token_count_pipeline()`의 인자를 수정하세요.

---

## 4) 데이터 간 중복 제거 (MinHash Dedup)

`build_dedup_pipeline()` / `run_dedup_pipeline()` 예시를 사용합니다.

일반적으로 2단계로 동작합니다.

1. **Signature 단계**: 각 문서의 MinHash 시그니처 생성
2. **Cluster/Filter 단계**: 시그니처 기반으로 유사 문서를 묶고 중복 제거

코드에서 기본 순서는 다음과 같습니다.

- `MinhashDedupSignature`로 시그니처 생성 (`./outputs/signatures`)
- `MinhashDedupCluster`로 클러스터링/중복 제거 후 결과 저장 (`./outputs/deduped`)

실행하려면 `pipeline_examples.py` 하단에서 아래 줄을 활성화하세요.

```python
# run_dedup_pipeline()
```

를

```python
run_dedup_pipeline()
```

로 바꾼 뒤 실행:

```bash
uv run python pipeline_examples.py
```

---

## 5) 실무 팁

- **샘플로 먼저 검증**: `train[:1000]`처럼 작게 시작한 뒤 범위를 늘리세요.
- **토큰 임계값 튜닝**: 도메인에 따라 `min_tokens`를 20~200 사이에서 탐색해보세요.
- **병렬도 조절**: `LocalPipelineExecutor(tasks=4)` 값을 CPU 코어 수에 맞게 조정하세요.
- **출력 분리**: 실험별로 `output_dir`/`logging_dir`를 분리하면 재현성이 좋아집니다.

---

## 6) 자주 발생하는 이슈

- **토크나이저 다운로드 오류**
  - 네트워크/권한 문제일 수 있습니다.
  - 사내망 환경이면 HF 미러/프록시 설정을 확인하세요.

- **메모리 부족(OOM)**
  - split 범위를 줄이거나 `tasks` 수를 낮추세요.

- **Dedup 결과가 과도하게 줄어듦**
  - 유사도/샤드 설정(라이브러리 옵션)과 전처리 기준을 점검하세요.

---

필요하면 다음 단계로, 이 예시를 Airflow/Prefect 같은 오케스트레이터에 연결하는 템플릿도 확장해드릴게요.
