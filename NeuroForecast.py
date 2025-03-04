import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import numpy as np

from GetData import get_stock_data
# Загрузка данных
df = get_stock_data("AAPL", "2024-01-01", "2024-05-01")

# Создание признаков
df['Lag1'] = df['Close'].shift(1)
df['Lag2'] = df['Close'].shift(2)
df['RollingMean'] = df['Close'].rolling(window=5).mean()
df.dropna(inplace=True)

# Масштабирование признаков
scaler = MinMaxScaler()
df[['Open', 'Close', 'Volume', 'Lag1', 'Lag2', 'RollingMean']] = scaler.fit_transform(df[['Open', 'Close', 'Volume', 'Lag1', 'Lag2', 'RollingMean']])

# Разделение данных на обучающую и тестовую выборки
X = df[['Open', 'Volume', 'Lag1', 'Lag2', 'RollingMean']]
y = df['Close']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Обучение модели
model = LinearRegression()
model.fit(X_train, y_train)

# Оценка модели
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

# Прогноз на следующую дату
last_data = df[['Open', 'Volume', 'Lag1', 'Lag2', 'RollingMean']].iloc[-1]
last_data_df = pd.DataFrame([last_data])
next_day_prediction = model.predict(last_data_df)
next_day_prediction = scaler.inverse_transform(np.concatenate([np.zeros((1, 3)), next_day_prediction.reshape(1, 1), np.zeros((1, 2))], axis=1))[0][3]

print(f'Prediction for the next day: {next_day_prediction}')