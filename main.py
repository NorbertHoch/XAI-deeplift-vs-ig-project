import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from model import LSTMModel
import matplotlib.pyplot as plt
from captum.attr import IntegratedGradients, DeepLift

def create_sequences(data, seq_len):
    X, y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i:i+seq_len])
        y.append(data[i+seq_len])
    return np.array(X), np.array(y)

# Data preproccesing

df = pd.read_csv('input/btcusd_1-min_data.csv')
df['date'] = pd.to_datetime(df['Timestamp'],unit='s').dt.date
group = df.groupby('date')
daily = group.agg(
    Open=('Open', 'first'),
    High=('High', 'max'),
    Low=('Low', 'min'),
    Close=('Close', 'last'),
    Volume=('Volume', 'sum')
).dropna().reset_index()

# Train test split

split = int(len(daily) * 0.8)
df_train = daily[:split]
df_test = daily[split:]

scaler = MinMaxScaler()
df_train = scaler.fit_transform(df_train[['Close']])
df_test = scaler.transform(df_test[['Close']])

X_train, y_train = create_sequences(df_train, 30)
X_test, y_test = create_sequences(df_test, 30)

# Training

X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.float32)
X_test_t  = torch.tensor(X_test,  dtype=torch.float32)
y_test_t  = torch.tensor(y_test,  dtype=torch.float32)

dataset = TensorDataset(X_train_t, y_train_t)
train_loader = DataLoader(dataset, batch_size=64, shuffle=True)

model = LSTMModel(input_size=1, hidden_size=64, num_layers=3, output_size=1)
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(50):
    for X_batch, y_batch in train_loader:
        output = model(X_batch)
        loss = loss_fn(output, y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# Evaluate on test set
model.eval()
with torch.no_grad():
    test_pred = model(X_test_t).numpy()
    test_loss = loss_fn(torch.tensor(test_pred), y_test_t).item()
    print(f"Test MSE loss: {test_loss:.6f}")

    # Inverse transform predictions and actuals
    pred_prices = scaler.inverse_transform(test_pred)
    actual_prices = scaler.inverse_transform(y_test_t.numpy())

# Plot predicted vs actual
fig, ax = plt.subplots(figsize=(14, 6))

ax.plot(actual_prices, label='Actual', alpha=0.8)
ax.plot(pred_prices, label='Predicted', alpha=0.8)
ax.set_xlabel('Day')
ax.set_ylabel('Price (USD)')
ax.set_title('Predicted vs Actual BTC Close Price (Test Set)')
ax.legend()

plt.tight_layout()
plt.show()

# Deeplift explanation

X_sample = X_test_t[:50]

dl = DeepLift(model)
dl_attributions = dl.attribute(X_sample, target=0)
dl_attributions = dl_attributions.detach().numpy()

fig, ax = plt.subplots(figsize=(14, 6))
ax.plot(dl_attributions.mean(axis=0), label='DeepLift', alpha=0.8)
ax.set_xlabel('Time Step')
ax.set_ylabel('Attribution')
ax.set_title('DeepLift Attribution (Test Set)')
ax.legend()
plt.tight_layout()
plt.show()

# IG explanation

ig = IntegratedGradients(model)
attributions, delta = ig.attribute(X_sample, target=0, return_convergence_delta=True)

attributions = attributions.detach().numpy()

fig, ax = plt.subplots(figsize=(14, 6))
ax.plot(attributions.mean(axis=0), label='Integrated Gradients', alpha=0.8)
ax.set_xlabel('Time Step')
ax.set_ylabel('Attribution')
ax.set_title('Integrated Gradients Attribution (Test Set)')
ax.legend()
plt.tight_layout()
plt.show()
