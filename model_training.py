import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
from feature_engineering import add_technical_indicators

class TimeSeriesDataset(Dataset):
    def __init__(self, features, targets, seq_len=60):
        self.X = features
        self.y = targets
        self.seq_len = seq_len
    def __len__(self): return len(self.X) - self.seq_len
    def __getitem__(self, idx):
        return (
            torch.tensor(self.X[idx:idx+self.seq_len], dtype=torch.float32),
            torch.tensor(self.y[idx+self.seq_len], dtype=torch.float32)
        )

class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)
    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        return self.fc(h_n[-1])

def train_model(csv_path, feature_cols, target_col='Close', epochs=10):
    df = pd.read_csv(csv_path, parse_dates=['Date'])
    df = add_technical_indicators(df)
    features = df[feature_cols].values
    targets = df[target_col].values
    dataset = TimeSeriesDataset(features, targets)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    model = LSTMModel(input_dim=len(feature_cols))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    for epoch in range(epochs):
        losses = []
        for X, y in loader:
            optimizer.zero_grad()
            preds = model(X).flatten()
            loss = criterion(preds, y)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        print(f"Epoch {epoch+1}, Loss: {np.mean(losses):.4f}")
    torch.save(model.state_dict(), 'models/lstm_model.pth')
