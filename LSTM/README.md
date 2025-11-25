
# NPP-LSTM (Bi-LSTM 기반 이상탐지)

원전 계측 시계열 데이터에 대해  
Bi-directional LSTM 기반으로 다음 시점 값을 예측하고, 예측 오차로 이상을 탐지하는 스크립트입니다.  
파일 이름: `npp-LSTM.py`

- Sliding window로 과거 구간(길이 95)을 입력으로 사용
- 마지막 시점 값(96번째)을 예측
- 예측값과 실제값의 절대 오차 → 이상 스코어로 사용
- 간단한 임계값(Threshold) 및 F1/Precision/Recall 계산


## 1. 환경 요구사항

- Python 3.8 이상
- 필수 라이브러리
  ```bash
  pip install \
    numpy \
    pandas \
    matplotlib \
    tensorflow \
    scikit-learn


> 스크립트는 `keras` / `tensorflow.keras`를 모두 사용하므로
> TensorFlow 2.x 환경에서 실행하는 것을 권장합니다.

---

## 2. 모델 구조 요약

* 입력 데이터

  * 정규화된 시계열 데이터 `(T, feature 수)`
  * 윈도우 길이: 96
  * 모델 입력: 앞 95개 시점(`win_given = 95`)
  * 타깃: 마지막 1개 시점(96번째)

* 전처리

  * 학습(normal) / 테스트(attack) 각각 최대 12,000 샘플 사용
  * 공통 min/max 기반 정규화 (train·test 통합 범위)
  * `time` 컬럼은 시간축, 나머지 컬럼은 feature로 사용

* 모델 (Bi-LSTM + 잔차 연결)

  * Input: `(win_given, n_features)`
  * BiLSTM(100, return_sequences=True) × 2층
  * BiLSTM(100) × 1층
  * Dense(n_features) → 예측값
  * 보조 입력(aux_input): 윈도우의 첫 시점 값
  * 최종 출력: 예측값 + aux_input (Skip connection 형태)
  * 손실 함수: MSE

* 이상 스코어

  * `|y_true - y_pred|`의 feature 평균 → anomaly score
  * 단순 이동 평균(10 포인트)도 계산 가능

---

## 3. 실행 방법

### 3.1 기본 실행

```bash
python npp-LSTM.py
```

스크립트 내부에서 다음 경로를 사용합니다.

* 학습(normal) 데이터

  * `../TranAD/data/train/rx/train.csv`
* 테스트(attack) 데이터

  * `../TranAD/data/test/rx/com/{data}.csv`
  * 코드 상단의 `data` 변수로 시나리오 이름 지정

    ```python
    data = "PZR-COM-01"   # 예: PZR-COM-01 시나리오
    ```

데이터 형식 가정:

* 공통 컬럼:

  * `time` (시간)
  * 나머지 센서 컬럼들 (연속형 수치)
* 학습/테스트 모두 `time` 컬럼 이름은 동일해야 합니다.

---

## 4. 주요 설정 변경 포인트

`npp-LSTM.py` 상단에서 아래 부분을 필요에 따라 수정할 수 있습니다.

```python
data = "PZR-COM-01"   # 테스트에 사용할 시나리오 CSV 이름 (확장자 제외)
win_given = 95        # 입력으로 사용할 시점 수
win_size = 96         # 전체 윈도우 길이
```

* 라벨 기준 시점(이상 구간 시작)은 코드 하단에서 설정합니다.

  ```python
  label = np.zeros(attack.shape[0])
  label[6000:] = 1    # 6000번째 인덱스 이후를 attack(1)으로 간주
  ```

* 임계값(Threshold)은 아래 부분에서 변경

  ```python
  predicted_table = check_anomaly(anomaly_score_test, 0.009)
  ```

---

## 5. 출력물

실행이 끝나면:

1. 학습 손실 플롯 (에폭별 loss)

   * 화면에 표시 (`plt.show()`)

2. 테스트 이상 스코어 플롯

   * `anomaly_score_test` 와 라벨(0/1)을 함께 그래프로 표시
   * 제목 예: `PZR-COM-01 Scenario Anomaly Detection Bi-LSTM`

3. Threshold 기반 이진 예측 결과 및 지표

   * `predicted_table` vs `label` 을 이용한
     F1, Precision, Recall, TP/TN/FP/FN 등이 콘솔에 출력

---

