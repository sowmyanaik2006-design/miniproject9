import matplotlib
matplotlib.use('Agg')  # <-- 1. Ye sabse upar add kiya
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

np.random.seed(42)
tf.random.set_seed(42)

DATA_FILE = "electricity_data.csv"
MODEL_FILE = "rnn_model.keras"

print("Checking data...") # <-- print add kiya taki pata chale chala

if not Path(DATA_FILE).exists():
    hours = np.arange(24 * 120)
    daily_pattern = 1.5 + 0.8 * np.sin(2 * np.pi * (hours - 6) / 24)
    weekly_pattern = 0.25 * np.sin(2 * np.pi * hours / (24 * 7))
    noise = np.random.normal(0, 0.15, len(hours))
    consumption = np.maximum(0.2, daily_pattern + weekly_pattern + noise)
    pd.DataFrame({"Hour": hours, "Consumption_kWh": consumption.round(3)}).to_csv(DATA_FILE, index=False)
    print("Sample electricity dataset created.")

data = pd.read_csv(DATA_FILE)
values = data["Consumption_kWh"].values.reshape(-1, 1)

if len(values) < 100:
    raise ValueError("Dataset is too small.")

split = int(len(values) * 0.8)
train_values = values[:split]
test_values = values[split - 24:]

scaler = MinMaxScaler()
train_scaled = scaler.fit_transform(train_values)
test_scaled = scaler.transform(test_values)

WINDOW = 24
def create_sequences(data, window):
    X, y = [], []
    for i in range(window, len(data)):
        X.append(data[i-window:i])
        y.append(data[i, 0])
    return np.array(X), np.array(y)

X_train, y_train = create_sequences(train_scaled, WINDOW)
X_test, y_test = create_sequences(test_scaled, WINDOW)

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(WINDOW, 1)),
    tf.keras.layers.SimpleRNN(32, activation="tanh"),
    tf.keras.layers.Dense(16, activation="relu"),
    tf.keras.layers.Dense(1)
])

model.compile(optimizer="adam", loss="mse", metrics=["mae"])
model.summary()

print("\nTraining started... 15 epochs")
model.fit(X_train, y_train, epochs=15, batch_size=32, validation_split=0.1, shuffle=False, verbose=1)

pred_scaled = model.predict(X_test, verbose=0)
actual = scaler.inverse_transform(y_test.reshape(-1, 1)).ravel()
predicted = scaler.inverse_transform(pred_scaled).ravel()

mae = mean_absolute_error(actual, predicted)
rmse = np.sqrt(mean_squared_error(actual, predicted))

print("\n--- RNN Evaluation ---")
print(f"MAE: {mae:.4f} kWh")
print(f"RMSE: {rmse:.4f} kWh")

model.save(MODEL_FILE)
print(f"Model saved as {MODEL_FILE}")

plt.figure(figsize=(10, 5))
plt.plot(actual[:168], label="Actual Consumption")
plt.plot(predicted[:168], label="Predicted Consumption")
plt.xlabel("Hour")
plt.ylabel("Electricity Consumption (kWh)")
plt.title("RNN Electricity Consumption Prediction")
plt.legend()
plt.tight_layout()
plt.savefig("prediction_graph.png")
# plt.show()  <-- 2. Isko hata diya / comment kiya
print("Graph saved as prediction_graph.png")
print("DONE!")