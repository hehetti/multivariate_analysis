import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# =========================
# 설정
# =========================
WINDOW_SIZE = 168
BATCH_SIZE = 64
NUM_EPOCHS = 100
LEARNING_RATE = 1e-3
PATIENCE = 10          # early stopping patience
MIN_DELTA = 1e-4       # 최소 개선량

MODEL_PATH = "catboost/best_cnn_lstm.pt"
SCALER_PATH = "catboost/scaler.pkl"

DATA_CSV = "catboost/final_dataset_complete.csv"  # 전체가 합쳐진 csv 파일 이름으로 바꿔줘

# scaler를 fit할 구간: 앞에서 (35064 - 168)번째 행까지
SCALER_FIT_END = 35064 - 168   # = 34896

# 마지막 8760개 샘플을 test set으로 사용
NUM_TEST_SAMPLES = 8760

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =========================
# Dataset / Model / EarlyStopping 정의
# =========================
class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(-1)  # (N, 1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class CNNLSTM(nn.Module):
    def __init__(self, num_features, cnn_channels=64, lstm_hidden=64,
                 lstm_layers=1, dropout=0.2):
        super().__init__()

        # Conv1d: (batch, channels, seq_len)
        self.conv1 = nn.Conv1d(
            in_channels=num_features,
            out_channels=cnn_channels,
            kernel_size=3,
            padding=1
        )
        self.bn1 = nn.BatchNorm1d(cnn_channels)
        self.pool = nn.MaxPool1d(kernel_size=2)

        # LSTM: 입력 차원은 conv 출력 채널 수
        self.lstm = nn.LSTM(
            input_size=cnn_channels,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True
        )

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(lstm_hidden, 1)

    def forward(self, x):
        # x: (batch, seq_len, num_features)
        x = x.transpose(1, 2)     # (batch, num_features, seq_len)

        x = self.conv1(x)         # (batch, cnn_channels, seq_len)
        x = self.bn1(x)
        x = torch.relu(x)
        x = self.pool(x)          # (batch, cnn_channels, seq_len')

        x = x.transpose(1, 2)     # (batch, seq_len', cnn_channels)

        lstm_out, (h_n, c_n) = self.lstm(x)
        last_hidden = h_n[-1]     # (batch, hidden)

        out = self.dropout(last_hidden)
        out = self.fc(out)        # (batch, 1)
        return out


class EarlyStopping:
    def __init__(self, patience=PATIENCE, min_delta=MIN_DELTA):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = np.inf
        self.counter = 0

    def step(self, val_loss, model):
        improved = val_loss < self.best_loss - self.min_delta

        if improved:
            self.best_loss = val_loss
            self.counter = 0
            torch.save(model.state_dict(), MODEL_PATH)
            print(f"  -> Validation loss improved. Model saved to {MODEL_PATH}")
        else:
            self.counter += 1
            print(f"  -> No improvement. EarlyStopping counter = {self.counter}/{self.patience}")

        return self.counter >= self.patience


# =========================
# 윈도우 생성 함수
# =========================
def make_sequences(X_matrix, y_array, window_size):
    """
    X_matrix: (M, num_features)
    y_array : (M,)  - 각 시점에 대응하는 타깃 y
    window_size: 168

    리턴:
        X_seq: (num_samples, window_size, num_features)
        y_seq: (num_samples,)
    """
    M = X_matrix.shape[0]
    X_list = []
    y_list = []

    # end_idx: 윈도우의 마지막 시간 인덱스
    for end_idx in range(window_size - 1, M):
        start_idx = end_idx - window_size + 1
        window = X_matrix[start_idx:end_idx + 1, :]   # (window_size, num_features)
        target = y_array[end_idx]                    # 마지막 시점의 y

        X_list.append(window)
        y_list.append(target)

    X_seq = np.stack(X_list, axis=0)
    y_seq = np.array(y_list)
    return X_seq, y_seq


# =========================
# 메인
# =========================
def main():
    # ---------- 1. 데이터 로드 ----------
    df = pd.read_csv(DATA_CSV)
    col_to_drop = df.columns[1]

    # 그 컬럼을 드롭
    df = df.drop(columns=col_to_drop)
    df = df.iloc[168:]  # 처음 168행 제거
    print("Original data shape:", df.shape)  # (N, num_cols)
    num_rows, num_cols = df.shape
    if SCALER_FIT_END > num_rows:
        raise ValueError(f"SCALER_FIT_END({SCALER_FIT_END})가 전체 행 개수({num_rows})보다 큼.")

    # ---------- 2. StandardScaler fit & transform ----------
    # 앞에서 (35064-168)번째까지의 행으로 fit
    fit_df = df.iloc[:SCALER_FIT_END]   # 0 ~ SCALER_FIT_END-1
    scaler = StandardScaler()
    scaler.fit(fit_df.values)

    # 전체 데이터에 스케일러 적용
    scaled_values = scaler.transform(df.values)            # (N, num_cols)
    joblib.dump(scaler, SCALER_PATH)
    print(f"Scaler fitted on first {SCALER_FIT_END} rows and saved to {SCALER_PATH}")

    # ---------- 3. y 생성 ----------
    # 첫 번째 열을 카피해서 첫 행만 빼고 y로 사용
    y_all = scaled_values[:, 0]        # (N,)
    y_all = y_all[1:]                  # (N-1,)
    M = y_all.shape[0]

    # ---------- 4. X 행렬 만들기 ----------
    # 첫 번째 열: 마지막 행 삭제 → (N-1, 1)
    col0 = scaled_values[:-1, 0:1]

    # 나머지 열: 첫 번째 행 삭제 → (N-1, num_cols-1)
    others = scaled_values[1:, 1:]     # row 1~N-1, col 1~end

    # 두 부분을 합쳐 X_matrix: (N-1, num_cols)
    X_matrix = np.concatenate([col0, others], axis=1)
    assert X_matrix.shape[0] == M, "X와 y의 길이가 맞지 않습니다."

    print("After shift, X_matrix shape:", X_matrix.shape)  # (N-1, num_cols)
    print("After shift, y_all shape    :", y_all.shape)     # (N-1,)

    # ---------- 5. 윈도우(168)로 시퀀스 생성 ----------
    X_seq, y_seq = make_sequences(X_matrix, y_all, WINDOW_SIZE)
    num_samples, T, F = X_seq.shape
    print("Sequence X shape:", X_seq.shape)  # (num_samples, 168, num_features)
    print("Sequence y shape:", y_seq.shape)  # (num_samples,)

    if num_samples <= NUM_TEST_SAMPLES:
        raise ValueError(f"윈도우로 만든 샘플 수({num_samples})가 NUM_TEST_SAMPLES({NUM_TEST_SAMPLES})보다 적음.")

    # ---------- 6. Train/Val/Test 분할 ----------
    # 마지막 8760개를 test set
    X_test = X_seq[-NUM_TEST_SAMPLES:]
    y_test = y_seq[-NUM_TEST_SAMPLES:]

    X_trainval = X_seq[:-NUM_TEST_SAMPLES]
    y_trainval = y_seq[:-NUM_TEST_SAMPLES]

    n_trainval = X_trainval.shape[0]
    n_val = n_trainval // 4  # 맨 처음 1/4을 validation

    X_val = X_trainval[:n_val]
    y_val = y_trainval[:n_val]

    X_train = X_trainval[n_val:]
    y_train = y_trainval[n_val:]

    print("Train samples:", X_train.shape[0])
    print("Val samples  :", X_val.shape[0])
    print("Test samples :", X_test.shape[0])

    # ---------- 7. Dataset / DataLoader ----------
    train_dataset = TimeSeriesDataset(X_train, y_train)
    val_dataset   = TimeSeriesDataset(X_val, y_val)
    test_dataset  = TimeSeriesDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False)
    test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False)

    # ---------- 8. 모델/손실함수/옵티마이저 ----------
    num_features = F
    model = CNNLSTM(num_features=num_features).to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    early_stopping = EarlyStopping(patience=PATIENCE, min_delta=MIN_DELTA)

    # ---------- 9. 학습 루프 (Early Stopping 포함) ----------
    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        train_losses = []

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())

        avg_train_loss = np.mean(train_losses)

        # ----- validation -----
        model.eval()
        val_losses = []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(DEVICE)
                y_batch = y_batch.to(DEVICE)

                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                val_losses.append(loss.item())

        avg_val_loss = np.mean(val_losses)
        print(f"[Epoch {epoch:03d}] Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}")

        # Early stopping 체크 (개선되면 모델 저장)
        stop = early_stopping.step(avg_val_loss, model)
        if stop:
            print("Early stopping triggered.")
            break

    # ---------- 10. 베스트 모델 로드 후 Test MSE ----------
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        print(f"Best model loaded from {MODEL_PATH}")
    else:
        print("Warning: MODEL_PATH not found. Using last epoch model.")

    model.eval()
    all_preds = []
    all_trues = []

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)

            outputs = model(X_batch)          # 스케일된 y 예측
            all_preds.append(outputs.cpu().numpy())
            all_trues.append(y_batch.cpu().numpy())

    # (N, 1) -> (N,)
    y_pred_scaled = np.concatenate(all_preds, axis=0).squeeze()
    y_true_scaled = np.concatenate(all_trues, axis=0).squeeze()

    # --- 여기서 스케일러에서 y에 해당하는 mean, std 가져오기 ---
    mean_y = scaler.mean_[0]
    std_y  = scaler.scale_[0]

    # --- 스케일 역변환 (원래 단위) ---
    y_pred_orig = y_pred_scaled * std_y + mean_y
    y_true_orig = y_true_scaled * std_y + mean_y

    # --- 원래 단위에서 MSE 계산 ---
    mse_orig = np.mean((y_pred_orig - y_true_orig) ** 2)
    print(f"Test MSE (original scale): {mse_orig:.6f}")


if __name__ == "__main__":
    main()
