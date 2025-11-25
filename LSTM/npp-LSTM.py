#!/usr/bin/env python
# coding: utf-8

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import precision_score, recall_score, f1_score
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Bidirectional, LSTM, Dense
from tensorflow.keras.optimizers import Adam


# =========================
# 1. 데이터 유틸 함수
# =========================

def load_csv(path: str) -> pd.DataFrame:
    """
    CSV 로드 + 컬럼 공백 제거.
    index_col 은 사용하지 않고, 모든 컬럼을 그대로 로드합니다.
    """
    df = pd.read_csv(path)
    df = df.rename(columns=lambda x: str(x).strip())
    return df


def normalize_minmax(train_df: pd.DataFrame, test_df: pd.DataFrame):
    """
    train/test 전체 범위를 기준으로 min-max 정규화.
    time 컬럼은 제외.
    """
    drop_cols = [c for c in ["time", "timestamp"] if c in train_df.columns or c in test_df.columns]

    train_feat = train_df.drop(columns=drop_cols, errors="ignore")
    test_feat = test_df.drop(columns=drop_cols, errors="ignore")

    # 전체 데이터 기준으로 min/max
    concat = pd.concat([train_feat, test_feat], axis=0)
    col_min = concat.min()
    col_max = concat.max()

    def _norm(df_feat):
        df_n = df_feat.copy()
        for c in df_feat.columns:
            mn, mx = col_min[c], col_max[c]
            if mx == mn:
                df_n[c] = df_feat[c] - mn
            else:
                df_n[c] = (df_feat[c] - mn) / (mx - mn)
        return df_n

    train_norm = _norm(train_feat)
    test_norm = _norm(test_feat)

    return train_norm, test_norm, col_min, col_max


def create_sliding_windows(data: np.ndarray, win_size: int, win_given: int):
    """
    data: (T, F)
    win_size: 전체 윈도우 길이 (예: 96)
    win_given: 입력으로 사용할 시점 수 (예: 95)

    반환:
      X: (N, win_given, F)
      y: (N, F)  - 윈도우 마지막 시점 값
    """
    T, F = data.shape
    X_list, y_list = [], []

    for start in range(0, T - win_size + 1):
        end = start + win_size
        window = data[start:end]         # (win_size, F)
        x = window[:win_given]          # (win_given, F)
        y = window[win_size - 1]        # (F,)
        X_list.append(x)
        y_list.append(y)

    X = np.stack(X_list) if X_list else np.empty((0, win_given, F))
    y = np.stack(y_list) if y_list else np.empty((0, F))
    return X, y


# =========================
# 2. Bi-LSTM 모델 정의
# =========================

def build_bilstm_model(n_features: int, win_given: int, lr: float = 1e-3) -> Model:
    """
    입력: (win_given, n_features)
    출력: 다음 시점의 feature 값 예측 (n_features)
    """
    inputs = Input(shape=(win_given, n_features))

    x = Bidirectional(LSTM(100, return_sequences=True))(inputs)
    x = Bidirectional(LSTM(100, return_sequences=True))(x)
    x = Bidirectional(LSTM(100))(x)

    outputs = Dense(n_features)(x)  # 마지막 시점 예측

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=Adam(learning_rate=lr), loss="mse")
    return model


# =========================
# 3. 이상탐지 로직
# =========================

def anomaly_scores(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """
    y_true, y_pred: (N, F)
    반환: (N,)  - 각 윈도우마다 feature 평균 절대 오차
    """
    return np.mean(np.abs(y_true - y_pred), axis=1)


def evaluate_threshold(scores: np.ndarray, labels: np.ndarray, threshold: float):
    """
    scores: (N,) anomaly score
    labels: (N,) 0/1
    threshold: score > threshold → 1 로 예측
    """
    preds = (scores > threshold).astype(int)
    precision = precision_score(labels, preds, zero_division=0)
    recall = recall_score(labels, preds, zero_division=0)
    f1 = f1_score(labels, preds, zero_division=0)
    return precision, recall, f1, preds


# =========================
# 4. 메인 파이프라인
# =========================

def main(args):
    # 4.1 데이터 로드
    print(f"[INFO] Loading train: {args.train}")
    print(f"[INFO] Loading test : {args.test}")
    train_raw = load_csv(args.train)
    test_raw = load_csv(args.test)

    # 길이 제한
    if args.max_len is not None and args.max_len > 0:
        train_raw = train_raw.iloc[:args.max_len]
        test_raw = test_raw.iloc[:args.max_len]

    # 4.2 정규화
    train_norm, test_norm, col_min, col_max = normalize_minmax(train_raw, test_raw)
    print("[INFO] train_norm shape:", train_norm.shape)
    print("[INFO] test_norm  shape:", test_norm.shape)

    train_np = train_norm.to_numpy().astype("float32")
    test_np = test_norm.to_numpy().astype("float32")

    # 4.3 슬라이딩 윈도우 생성
    win_size = args.win_size
    win_given = args.win_given

    X_train, y_train = create_sliding_windows(train_np, win_size, win_given)
    X_test, y_test = create_sliding_windows(test_np, win_size, win_given)

    print("[INFO] X_train / y_train shape:", X_train.shape, y_train.shape)
    print("[INFO] X_test  / y_test  shape:", X_test.shape, y_test.shape)

    if X_train.shape[0] == 0 or X_test.shape[0] == 0:
        raise RuntimeError("윈도우 생성 결과가 비어 있습니다. win_size / 데이터 길이를 확인하세요.")

    n_features = X_train.shape[-1]

    # 4.4 모델 생성 및 학습
    model = build_bilstm_model(n_features, win_given, lr=args.lr)
    model.summary(print_fn=lambda x: print("[MODEL] " + x))

    os.makedirs(f"model/{args.name}", exist_ok=True)
    ckpt_path = f"model/{args.name}/bilstm.h5"

    if args.mode in ("train", "train_test"):
        print("[INFO] Start training...")
        history = model.fit(
            X_train,
            y_train,
            epochs=args.epochs,
            batch_size=args.batch_size,
            validation_split=0.1,
            verbose=1,
        )
        model.save(ckpt_path)
        print(f"[INFO] Model saved to {ckpt_path}")

        # 학습 loss 플롯
        os.makedirs(f"plots/{args.name}", exist_ok=True)
        plt.figure()
        plt.plot(history.history["loss"], label="loss")
        if "val_loss" in history.history:
            plt.plot(history.history["val_loss"], label="val_loss")
        plt.xlabel("Epoch")
        plt.ylabel("MSE loss")
        plt.legend()
        plt.title(f"Training Loss - {args.name}")
        plt.savefig(f"plots/{args.name}/training_loss.png")
        plt.close()

    # 4.5 테스트 (항상 또는 별도)
    if args.mode in ("test", "train_test"):
        if args.mode == "test":
            # 이미 학습된 모델이 있다고 가정하고 로드
            if os.path.exists(ckpt_path):
                print(f"[INFO] Loading model from {ckpt_path}")
                model = tf.keras.models.load_model(ckpt_path)
            else:
                raise FileNotFoundError(f"{ckpt_path} 가 존재하지 않습니다. 먼저 --mode train 또는 train_test 로 학습하세요.")

        print("[INFO] Inference on train/test for anomaly scores...")

        # train reconstruction error (threshold 설정용)
        y_train_pred = model.predict(X_train, batch_size=args.batch_size, verbose=0)
        scores_train = anomaly_scores(y_train, y_train_pred)

        # test reconstruction error
        y_test_pred = model.predict(X_test, batch_size=args.batch_size, verbose=0)
        scores_test = anomaly_scores(y_test, y_test_pred)

        # 4.5.1 라벨 생성 (간이: anomaly_start 이후를 1로)
        n_test = scores_test.shape[0]
        labels = np.zeros(n_test, dtype=int)
        if 0 <= args.anomaly_start < n_test:
            labels[args.anomaly_start:] = 1

        # 4.5.2 threshold 설정
        if args.threshold is not None:
            threshold = args.threshold
            print(f"[INFO] Using user-defined threshold: {threshold}")
        else:
            # train score의 상위 q 분위수 사용
            q = args.th_quantile
            threshold = float(np.quantile(scores_train, q))
            print(f"[INFO] Using train quantile threshold (q={q}): {threshold}")

        precision, recall, f1, preds = evaluate_threshold(scores_test, labels, threshold)

        print("[RESULT] Bi-LSTM Anomaly Detection")
        print(f"  Threshold : {threshold:.6f}")
        print(f"  Precision : {precision:.4f}")
        print(f"  Recall    : {recall:.4f}")
        print(f"  F1-score  : {f1:.4f}")

        # 4.5.3 스코어 + 라벨 플롯
        os.makedirs("plots", exist_ok=True)
        plt.figure(figsize=(12, 4))
        plt.plot(scores_test, label="Anomaly Score")
        plt.plot(labels * scores_test.max(), label="Label (scaled)", alpha=0.5)
        plt.axhline(threshold, color="red", linestyle="--", label="Threshold")
        plt.xlabel("Window index")
        plt.ylabel("Score")
        plt.title(f"{args.name} Scenario Anomaly Detection Bi-LSTM")
        plt.legend()
        out_plot = f"plots/{args.name}_bilstm_score.png"
        plt.savefig(out_plot)
        plt.close()
        print(f"[INFO] Score plot saved to {out_plot}")


# =========================
# 5. CLI entry
# =========================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Bi-LSTM 기반 NPP 시계열 이상탐지 (npp-LSTM)"
    )
    parser.add_argument(
        "--train",
        required=True,
        help="학습용 CSV 경로",
    )
    parser.add_argument(
        "--test",
        required=True,
        help="테스트용 CSV 경로",
    )
    parser.add_argument(
        "--name",
        default="NPP_LSTM",
        help="실험 이름 / 모델·플롯 저장 폴더 이름 (기본: NPP_LSTM)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="학습 epoch 수 (기본 10)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=128,
        help="배치 크기 (기본 128)",
    )
    parser.add_argument(
        "--win_size",
        type=int,
        default=96,
        help="전체 윈도우 길이 (기본 96)",
    )
    parser.add_argument(
        "--win_given",
        type=int,
        default=95,
        help="입력으로 사용할 시점 수 (기본 95, win_size-1 권장)",
    )
    parser.add_argument(
        "--max_len",
        type=int,
        default=12000,
        help="train/test 앞에서부터 사용할 최대 길이 (0 또는 음수면 전체 사용)",
    )
    parser.add_argument(
        "--anomaly_start",
        type=int,
        default=6000,
        help="이 인덱스부터 라벨 1(이상)으로 취급 (기본 6000)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="학습률 (기본 1e-3)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="임계값 (직접 지정, None이면 train quantile 사용)",
    )
    parser.add_argument(
        "--th_quantile",
        type=float,
        default=0.99,
        help="threshold 미지정 시, train score 상위 quantile (기본 0.99)",
    )
    parser.add_argument(
        "--mode",
        choices=["train", "test", "train_test"],
        default="train_test",
        help='"train" / "test" / "train_test" (기본 train_test)',
    )

    args = parser.parse_args()
    if args.max_len is not None and args.max_len <= 0:
        args.max_len = None

    main(args)
