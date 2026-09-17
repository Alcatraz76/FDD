# 예제 실행 환경 준비

학습 실행 환경(컨테이너)은 **플랫폼에서 사용자가 직접 생성**합니다.
그래서 예제 폴더에는 `requirements.txt` 파일이 없으며, 코드파일과 함께 업로드하지도 않습니다.
필요한 패키지는 「컨테이너」 탭의 「설치 패키지」 편집창에 적습니다.

학습 코드는 외부망이 막힌 환경에서 실행되므로, 학습이 도는 중에는 패키지를 설치할 수 없습니다.

---

## 컨테이너 생성 절차

프로젝트 상세의 「컨테이너」 탭에서 진행하며, 카드 3개가 위에서 아래로 순서대로 의존합니다.

| 순서 | 카드 | 하는 일 | 상태값 |
|------|------|--------|--------|
| 1 | 프로비저닝 | 프로젝트 전용 학습 환경 할당. 「실패」인 경우에만 「재시도」 버튼이 표시됩니다 | 대기 중 · 준비 중 · 성공 · 실패 |
| 2 | 설치 패키지 | 편집창에 패키지 목록을 적고 「설치」 → 이미지 빌드 후 컨테이너 자동 실행 | 대기 중 · 설치 중 · 설치 완료 · 설치 실패 |
| 3 | 컨테이너 | `OFF` / `ON` 스위치로 실행 환경을 켜고 끕니다 | 비 활성 · 시동 중 · 활성 · 종료 중 · 상태 미확인 |

- 「프로비저닝」이 **성공**해야 「설치」를 누를 수 있고, 설치가 **완료**돼야 컨테이너를 실행할 수 있습니다.
- 「설치」는 이미지 빌드가 포함된 무거운 작업이라 시간이 걸립니다.
- 컨테이너를 끄면 그 시점에 동작 중인 연구가 취소됩니다.
- 코드파일 탭에 올리는 필수 파일은 `model.py` · `data.py` · `train_eval.py` 3개입니다. `requirements.txt` 는 올리지 않습니다.

---

## 플랫폼 기본 제공 이미지 (사용자 설치 불필요)

컨테이너는 플랫폼이 제공하는 기본 이미지 위에 만들어집니다.
아래 패키지는 **이미 설치되어 있으므로 「설치 패키지」에 적지 않습니다**.

| 구분 | 내용 |
|------|------|
| 파이썬 | 3.12.3 |
| 연합학습 | `nvflare==2.7.0` |
| 데이터·설정 | `numpy` · `pandas` · `omegaconf` |
| 저장소·전송 | `awscli` · `boto3` · `botocore` · `cloudpathlib` |
| DB·큐 | `mysql-connector-python==9.7.0` · `psycopg2` · `redis` · `celery` |
| 기타 | `pyopenssl==26.2.0` · `tensorboard` · `bitsandbytes` |

> `torch` 는 기본 이미지에 없습니다. 사용하는 예제에 맞춰 「설치 패키지」에 직접 적어야 합니다.

「설치 패키지」에 적은 내용은 학습 컨테이너 이미지를 만들 때 다음 명령으로 설치됩니다.

```bash
uv pip install -r requirements.txt --system --index-strategy unsafe-best-match
```

---

## 예제별 추가 패키지

| 예제 | 「설치 패키지」에 적을 패키지 |
|------|---------------------------|
| `mlp` · `mlp_prox` · `mlp_scaffold` · `cnn` | `torch` · `torchvision` |
| `llm_peft` | `torch` · `transformers` · `peft` |

작성 예시 — 저장소를 지정하는 방식 (CUDA 12.6)

```
--extra-index-url https://download.pytorch.org/whl/cu126
torch==2.6.0+cu126
torchvision==0.21.0+cu126
```

작성 예시 — 개별 패키지 URL 방식

```
torch @ https://download.pytorch.org/whl/cu126/torch-2.6.0%2Bcu126-cp312-cp312-linux_x86_64.whl
torchvision @ https://download.pytorch.org/whl/cu126/torchvision-0.21.0%2Bcu126-cp312-cp312-linux_x86_64.whl
```

- 편집창은 파일을 올리는 자리가 아니라 `requirements.txt` 의 내용을 그대로 적는 자리입니다.
- 저장소 주소(`--extra-index-url`, `-f`)는 맨 위에 적습니다.
- 학습에 필요한 패키지만 남겨 목록을 간소화하고, 버전을 명시합니다.
- 문법 오류나 설치 가능 여부를 미리 검사하지 않습니다. 개별 환경에서 `uv pip install -r requirements.txt` 로 설치를 시험한 뒤 「설치」를 누르십시오.

---

## 실행환경을 구성할 수 없는 경우

1. 패키지 목록과 별도로 특정 파일을 다운로드해 직접 설치하는 bash shell 이 필요한 경우
2. 로컬 파일을 참조하는 경우 — `package @ file:///myfile...`
3. 패키지 종속성으로 인해 `uv pip install` 한 번으로 설치할 수 없는 경우

상세한 작성 규칙과 예시는 [연합학습 가이드](https://fdd.k-melloddy.com/fl-guide) 를 확인하세요.
