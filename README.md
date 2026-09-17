# K-MELLODDY FDD 예제

FDD 연합학습 플랫폼에 등록할 학습 코드 예제 모음입니다.

## 가이드 문서

| 문서 | 주소 | 내용 |
|------|------|------|
| 플랫폼 사용가이드 | <https://fdd.k-melloddy.com/user-guide> | 화면·탭별 사용 방법 |
| 연합학습 가이드 | <https://fdd.k-melloddy.com/fl-guide> | 학습 코드 작성, 실행환경 준비, 연합학습 주의사항 |

## 예제 구성

| 경로 | 모델 | 집계 알고리즘 | 비고 |
|------|------|-------------|------|
| `examples/mlp/` | MLP | fedavg | 기본 참조 구현 |
| `examples/cnn/` | CNN | fedavg | |
| `examples/mlp_prox/` | MLP | fedprox | `controller_manager.py` 사용 |
| `examples/mlp_scaffold/` | MLP | scaffold | `controller_manager.py` 사용 |
| `examples/llm_peft/` | LLM + PEFT(LoRA) | — | `config.yaml` 포함 |
| `examples/script/` | — | — | 플랫폼이 job 생성 시 자동으로 추가하는 파일 (참고용) |

## 실행 환경

학습 컨테이너는 플랫폼의 「컨테이너」 탭에서 만듭니다. 예제에 `requirements.txt` 파일은 포함하지 않으며 업로드 대상도 아닙니다.

준비 절차와 기본 제공 패키지 목록은 [examples/README.md](examples/README.md) 를 확인하세요.
