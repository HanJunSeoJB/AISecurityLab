# TranAD (Transformer-based Anomaly Detection)

원전 계측 등 시계열 센서 데이터에 대해  
Transformer 기반 Autoencoder(TranAD)로 이상탐지를 수행하는 스크립트입니다.  
파일 이름: `TranAD_final.py`

- 시계열 센서 데이터를 슬라이딩 윈도우로 변환
- Transformer Encoder–Decoder 기반 2단계 재구성(Phase 1/2)
- 재구성 오차 기반 이상 스코어 산출
- POT(SPOT) 기반 임계값 추정 및 지표 계산



## 1. 환경 요구사항

- Python 3.8 이상
- 필수 라이브러리
  ```bash
  pip install \
    numpy \
    pandas \
    matplotlib \
    torch \
    scikit-learn \
    tqdm

> `torch`는 CUDA 환경에 따라 설치 방법이 다릅니다.
> PyTorch 공식 사이트 가이드에 맞춰 설치하는 것을 권장합니다.

---

## 2. 모델 구조 요약

* 입력: 정규화된 시계열 `(T, feature 수)`
* 슬라이딩 윈도우: `(T, 윈도우길이, feature 수)` 로 변환
* 모델

  * Positional Encoding + Transformer Encoder
  * Transformer Decoder 두 단계

    * Phase 1: 기본 재구성
    * Phase 2: Phase 1 재구성 오차를 조건으로 추가 재구성
  * 최종 출력: 마지막 시점의 feature 재구성값
* 손실: MSE (입력 vs 재구성)
* 이상 스코어: 재구성 오차의 평균 또는 feature별 오차
* 임계값 및 지표: `pot_eval`(POT) + Hit@k, NDCG@k 등

---

## 3. 실행 방법

### 3.1 기본 실행 예시

```bash
python TranAD_final.py \
  --train ./data/train/rx/train.csv \
  --test  ./data/test/rx/com/PZR-COM-01.csv \
  --name PZR_COM_01_TranAD \
  --epochs 5 \
  --win_size 96 \
  --max_len 12000 \
  --anomaly_start 6000 \
  --mode train_test
```

* `--mode train_test` : 학습 후 테스트까지 수행
* `--mode train`      : 학습만 수행
* `--mode test`       : (동일 코드 내에서) 테스트만 수행

> CSV에 `time` 컬럼이 있어도 되지만, 학습에는 사용하지 않으며 자동으로 제외됩니다.
> 나머지 컬럼들은 모두 feature로 사용됩니다.

---

## 4. 커맨드라인 인자 설명

```text
--train         : 학습용 CSV 경로
--test          : 테스트용 CSV 경로
--name          : 실험 이름 / 모델·플롯 저장 폴더 이름 (기본: TranAD)
--epochs        : 학습 epoch 수 (기본 5)
--win_size      : 슬라이딩 윈도우 길이 (기본 96)
--max_len       : train/test 모두 앞에서부터 사용할 최대 길이
                  0 또는 음수면 전체 사용 (기본 12000)
--anomaly_start : 테스트 시퀀스에서 이 인덱스부터 라벨 1(이상)으로 취급 (기본 6000)
--mode          : "train" / "test" / "train_test"
                  - train      : 학습만
                  - test       : (동일 프로세스 내) 테스트만
                  - train_test : 학습 후 테스트까지
```

---

## 5. 출력물

실행이 끝나면:

1. 모델 체크포인트 저장

   * 경로: `./model/{name}/{name}.ckpt`

2. 학습 손실 및 학습률 그래프

   * 경로: `plots/{name}/training-graph.pdf`

3. 테스트 이상 스코어 플롯

   * 경로: `plots/{name}_score_plot.png`
   * 내용: 평균 재구성 오차(Anomaly Score) + 라벨(0/1)

4. 콘솔 출력

   * POT 기반 F1, Precision, Recall 등 결과
   * Hit@k, NDCG@k 등의 지표

---

