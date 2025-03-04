import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import numpy as np

from GetData import get_stock_data

def predict(df):
    df.dropna(inplace=True)
    # Создание признаков
    features = ['Open', 'volume', 'moving_average_20', 'exponential_moving_average_20', 'rsi', 'volatility',
                'percentage_change', 'macd', 'macd_signal', 'bollinger_upper', 'bollinger_lower']
    X = df[features]
    y = df['close']

    # Масштабирование признаков
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)

    # Разделение данных на обучающую и тестовую выборки
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Обучение модели
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Оценка модели
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f'Mean Squared Error: {mse}')

    # Выбор последних данных
    last_data = df[features].iloc[-1].to_frame().T

    # Масштабирование данных
    last_data_scaled = scaler.transform(last_data)

    # Прогноз на следующий день
    next_day_prediction = model.predict(last_data_scaled)[0]

    print(f'Prediction for the next day: {next_day_prediction}')