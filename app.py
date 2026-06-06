import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

print("Downloading stock data...")

stock = yf.download("AAPL", start="2020-01-01", end="2024-01-01")

stock['Prediction'] = stock[['Close']].shift(-1)

X = np.array(stock[['Close']])[:-1]
y = np.array(stock['Prediction'])[:-1]

X_train, X_test, y_train, y_test = train_test_split(
X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

predictions = model.predict(X_test)

accuracy = model.score(X_test, y_test)

print(f"Model Accuracy: {accuracy:.4f}")

future_price = model.predict([[stock['Close'].iloc[-1]]])

print(f"Predicted Next Day Price: ${future_price[0]:.2f}")

plt.figure(figsize=(10, 5))
plt.plot(y_test, label="Actual Price")
plt.plot(predictions, label="Predicted Price")
plt.title("Stock Price Prediction")
plt.xlabel("Samples")
plt.ylabel("Price")
plt.legend()
plt.show()
