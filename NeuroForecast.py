import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import numpy as np

from GetData import get_stock_data

def predict_stock_price(df):
    """
    Прогнозирует цену акций на следующий день на основе обученной модели.

    :param df: DataFrame с данными акций и признаками
    :return: Прогнозируемая цена акций на следующий день
    """
    # Удаление строк с пропущенными значениями из DataFrame
    df.dropna(inplace=True)
    
    # Определение признаков (X) и целевой переменной (y)
    features = ['Open', 'Volume', 'moving_average_20', 'exponential_moving_average_20', 
                'rsi', 'volatility', 'percentage_change', 'macd', 'macd_signal', 
                'bollinger_upper', 'bollinger_lower']
    
    # Выделение признаков для обучения модели
    X = df[features]
    # Выделение целевой переменной (цена закрытия акций)
    y = df['Close']

    # Масштабирование признаков для улучшения сходимости модели
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    # Разделение данных на обучающую (80%) и тестовую (20%) выборки
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Создание экземпляра модели линейной регрессии
    model = LinearRegression()
    # Обучение модели на обучающих данных
    model.fit(X_train, y_train)

    # Прогнозирование цен на тестовой выборке
    y_pred = model.predict(X_test)
    # Оценка качества модели с использованием среднеквадратичной ошибки
    mse = mean_squared_error(y_test, y_pred)
    print(f'Mean Squared Error: {mse}')

    # Выбор последних данных для прогнозирования следующей цены
    last_data = df[features].iloc[-1].to_frame().T

    # Масштабирование последних данных с использованием того же скейлера
    last_data_scaled = scaler.transform(last_data)

    # Прогнозирование цены акций на следующий день
    next_day_prediction = model.predict(last_data_scaled)[0]

    # Вывод прогнозируемой цены на следующий день
    print(f'Prediction for the next day: {next_day_prediction}')
