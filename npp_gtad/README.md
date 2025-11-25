# NPP-GTAD (Graph-based Temporal Anomaly Detection)

원전 계측 데이터에 대해  
Edge-aware GAT + TCN 기반의 이상탐지를 수행하는 스크립트입니다.  
파일 이름: `g2_gpt2.py` (내부 주석: NPP-GTAD)

- 시계열 센서 데이터를 그래프로 구성
- Pearson 상관계수 + 마지막 시점 차이로 edge feature 구성
- 슬라이딩 윈도우 후 GAT + TCN으로 이진 분류 (정상/이상)
- 데이터 컬럼이 **실제 태그명** 이든, **익명 ID(v001~v127)** 이든 모두 사용 가능



## 1. 환경 요구사항

- Python 3.8 이상
- 필수 라이브러리
  ```bash
  pip install \
    numpy \
    pandas \
    matplotlib \
    seaborn \
    torch \
    torch-geometric'


> `torch` / `torch-geometric`는 CUDA 버전에 따라 설치 방법이 다를 수 있으니
> PyTorch 공식 사이트 가이드에 맞춰 설치하는 것을 권장합니다.

---

## 2. 모델 구조 요약

* 입력: `(배치, 윈도우길이, feature 수)`
* 슬라이딩 윈도우 생성
* 그래프 구성

  * 노드: 태그(feature)
  * 엣지: fully-connected (또는 그룹별 fully-connected)
  * 엣지 feature:

    * Pearson 상관계수 (train 기준)
    * 마지막 시점의 값 차이
* 모델

  * TCN(Conv1d)으로 시간 축 특징 추출
  * Edge-aware GAT (`EdgeAwareGATConv`)로 그래프 특징 추출
  * Linear + BN + ReLU + Dropout → 2-class logits
* 손실: Cross-Entropy
* 옵티마이저: Adam

---

## 3. 실행 방법

### 3.1 기본 실행 예시 (원본 태그 사용)

```bash
python g2_gpt2.py \
  --train ./train/rx/train.csv \
  --test  ./test/rx/test.csv \
  --model ./best_model.pt \
  --win_size 96 \
  --epochs 100 \
  --patience 10
```

## 4. 커맨드라인 인자 설명

```text
--train    : 학습용 CSV 경로 (index_col=0)
--test     : 테스트용 CSV 경로 (index_col=0)
--model    : best 모델을 저장할 .pt 파일 경로
--win_size : 슬라이딩 윈도우 길이 (기본 96)
--epochs   : 최대 epoch 수 (기본 100)
--patience : validation 개선이 없을 때 조기 종료 epoch 수 (기본 10)
```

---

## 5. 출력물

실행이 끝나면:

1. `--model` 인자로 지정한 경로에 best 모델 저장

   * 예: `best_model.pt`
2. 학습/검증 로그 및 최종 성능 지표 콘솔 출력
3. 테스트 스코어 & attack 라벨 시각화 PNG 저장

   * 예: `test_score_attack_label.png`

---

