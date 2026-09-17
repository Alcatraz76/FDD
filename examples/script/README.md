# 플랫폼 표준 스크립트 (참고용)

「코드파일」 탭에 올린 사용자 코드와 함께, 플랫폼이 연합학습 job 을 만들 때 **자동으로 넣는** 파일입니다.
사용자가 준비하거나 업로드하지 않습니다.

- 같은 이름의 파일을 「코드파일」 탭에 올리면 플랫폼이 넣는 파일과 충돌합니다.
- 항상 플랫폼 기준 최신 버전으로 덮어써지므로, 이 사본을 수정해도 학습에는 반영되지 않습니다.
- 이 폴더의 파일은 동작을 이해하기 위한 참고용입니다. 미제공 모듈 2개와 집계서버·KDA 연결이 필요하므로 그대로 실행되지 않습니다.
- 각 예제 폴더(`mlp` · `cnn` 등)에 들어 있는 `script.py` · `controller_manager.py` 도 같은 사본입니다.

---

## 표준 스크립트 4개

| 파일 | 역할 | 저장소 포함 |
|------|------|-----------|
| `script.py` | 클라이언트에서 연합학습을 진행하는 실행 스크립트 | ● |
| `controller_manager.py` | 「FEDPROX」 · 「SCAFFOLD」 를 고른 경우에 필요한 계산 제공 | ● |
| `training_manager.py` | 상태·지표 관리, KDA 데이터 수신, 에러 보고 | ✕ 보안상 미제공 |
| `model_manager.py` | 사전학습모델 다운로드(`download_model()`), 모델 통계 출력 | ✕ 보안상 미제공 |

미제공 2개도 학습 시에는 사용자 코드와 같은 자리에 배포되므로 `import` 만 하면 쓸 수 있습니다.
사용 방법은 [연합학습 가이드](https://fdd.k-melloddy.com/fl-guide) 의 「VI. 코드 예시 — model_manager」 에 정리되어 있습니다.

---

## `script.py`

집계서버 연결, Round 반복, 글로벌 모델 수신과 전송, 진행 상태와 지표 보고를 모두 이 스크립트가 합니다.
사용자 코드는 이 흐름을 만들지 않고, 정해진 자리에서 호출되는 부품으로 동작합니다.

1. 사용자 업로드 파일 import — `model.py` · `data.py` · `train_eval.py`
2. `flare.init()` → `TrainingManager` 초기화
3. `set_config()` — `config.yaml` 을 올린 경우에만 로드 (없으면 `configs = None`)
4. `data_download()` — KDA DB 에서 원천 데이터 수신
5. `data_loader(raw_datas, configs)` → `model_loader(configs)`
6. Round 반복 — `flare.receive()` → 글로벌 모델 `evaluate()` → `train()` → 로컬 `evaluate()` → `flare.send()`

- 학습·평가 전에 모델을 GPU(`cuda:0`)로 옮기는 것도 이 스크립트가 합니다.
- 파일명과 함수명을 고정된 이름 그대로 import 하므로, **파일명 · 함수명 · 인자 순서 · 반환 형태**를 맞춰야 합니다.
- 각 단계는 상태값으로 보고되며, 예외가 나면 그 시점의 상태와 traceback 이 학습 로그에 남습니다.

---

## `controller_manager.py`

집계 알고리즘으로 「FEDPROX」 또는 「SCAFFOLD」 를 고른 경우에만 사용합니다.
Prox 항 계산, control variate 관리, 글로벌 모델과의 차이 계산이 모두 이 모듈에 들어 있습니다.

| 클래스 | 쓰는 때 | 하는 일 |
|--------|--------|--------|
| `FLFedProx` | 「FEDPROX」 | 글로벌 모델과 멀어진 정도를 손실에 더합니다 — `fedprox_apply(model, loss)` |
| `FLScaffold` | 「SCAFFOLD」 | 기관별 기울기 편차(control variate)를 관리하고 서버로 보낼 값을 만듭니다 |

- 두 클래스는 **싱글턴** 입니다. 같은 라운드 안에서 몇 번 만들어도 같은 객체가 돌아옵니다.
- `FLScaffold` 호출 순서 — `scaffold_apply` 는 `optimizer.step()` **앞**, `scaffold_update` 는 모든 epoch 이 끝난 뒤 **한 번**.
- `train_eval.py` 작성 예시는 연합학습 가이드 「train_eval.py — 집계 알고리즘에 따른 변형」 에 있습니다.

---

## 가이드 문서

| 문서 | 주소 |
|------|------|
| 연합학습 가이드 | <https://fdd.k-melloddy.com/fl-guide> |
| 플랫폼 사용가이드 | <https://fdd.k-melloddy.com/user-guide> |
