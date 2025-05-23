#pip install TA-Lib

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
# import talib

def get_stock_data(ticker: str, start_date: str, end_date: str):
    stock = yf.Ticker(ticker)
    data = stock.history(start=start_date, end=end_date)
    df = data[['Open', 'Close', 'High', 'Low', 'Volume']].reset_index()
    df.columns = ['Date', 'Open', 'Close', 'High', 'Low', 'Volume']
    return df

df = get_stock_data("AAPL", "2024-01-01", "2024-02-01")
df.set_index('Date', inplace=True)
#df.rename(columns={'Close': 'close', 'Volume': 'volume'}, inplace=True)

# 1. Объём продаж (volume)
def calculate_volume(df):
    return df['Volume']

# 2. Скользящие средние (Moving Average)
def moving_average(df, window):
    return df['Close'].rolling(window=window).mean()

# 3. Экспоненциальные скользящие средние (Exponential Moving Average)
def exponential_moving_average(df, window):
    return df['Close'].ewm(span=window, adjust=False).mean()

# 4. Индекс относительной силы (RSI)
def calculate_rsi(df, window=14):
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# 5. Уровни Фибоначчи
def fibonacci_levels(df):
    max_price = df['Close'].max()
    min_price = df['Close'].min()
    diff = max_price - min_price
    level1 = max_price - 0.236 * diff
    level2 = max_price - 0.382 * diff
    level3 = max_price - 0.618 * diff
    return {
        'level1': level1,
        'level2': level2,
        'level3': level3,
        'max_price': max_price,
        'min_price': min_price
    }

# 6. Стандартное отклонение (Volatility)
def calculate_volatility(df, window=20):
    return df['Close'].rolling(window=window).std()

# 7. Процент изменения (Percentage Change)
def calculate_percentage_change(df):
    return df['Close'].pct_change() * 100

# 8. MACD
def calculate_macd(df):
    ema12 = exponential_moving_average(df, 12)
    ema26 = exponential_moving_average(df, 26)
    macd = ema12 - ema26
    return macd

# 9. MACD Signal Line
def calculate_macd_signal(df, window=9):
    macd = calculate_macd(df)
    return macd.rolling(window=window).mean()

# 10. Bollinger Bands
def calculate_bollinger_bands(df, window=20, num_std_dev=2):
    ma = moving_average(df, window)
    rolling_std = df['Close'].rolling(window).std()
    upper_band = ma + (rolling_std * num_std_dev)
    lower_band = ma - (rolling_std * num_std_dev)
    return upper_band, lower_band

# 11. Стохастический осциллятор (Stochastic Oscillator)
def calculate_stochastic_oscillator(df, window=14):
    high14 = df['High'].rolling(window=window).max()
    low14 = df['Low'].rolling(window=window).min()
    stochastic = 100 * (df['Close'] - low14) / (high14 - low14)
    return stochastic

# 12. Свечные паттерны
# def calculate_candlestick_patterns(df):
#     patterns = {
#         'Bullish Engulfing': talib.CDLENGULFING(df['Open'], df['High'], df['Low'], df['Close']),
#         'Hammer': talib.CDLHAMMER(df['Open'], df['High'], df['Low'], df['Close']),
#         'Shooting Star': talib.CDLSHOOTINGSTAR(df['Open'], df['High'], df['Low'], df['Close'])
#     }
#     return patterns

# Расчет метрик
def calculateMetrics(df):
    df['Volume'] = calculate_volume(df)
    df['moving_average_20'] = moving_average(df, window=20)
    df['exponential_moving_average_20'] = exponential_moving_average(df, window=20)
    df['rsi'] = calculate_rsi(df)
    df['volatility'] = calculate_volatility(df)
    df['percentage_change'] = calculate_percentage_change(df)
    df['macd'] = calculate_macd(df)
    df['macd_signal'] = calculate_macd_signal(df)
    df['bollinger_upper'], df['bollinger_lower'] = calculate_bollinger_bands(df)
    df['stochastic'] = calculate_stochastic_oscillator(df)
    # candlestick_patterns = calculate_candlestick_patterns(df)

    fib_levels = fibonacci_levels(df)

    # Вывод результатов
    print(df.tail())
    print("Уровни Фибоначчи:", fib_levels)
    # print("Свечные паттерны:", candlestick_patterns)

    # Визуализация
    plt.figure(figsize=(14, 12))

    # График цен и скользящих средних
    plt.subplot(4, 1, 1)
    plt.plot(df.index, df['Close'], label='Цена закрытия', color='blue')
    plt.plot(df.index, df['moving_average_20'], label='20-дневная скользящая средняя', color='orange')
    plt.plot(df.index, df['exponential_moving_average_20'], label='20-дневная экспоненциальная скользящая средняя', color='green')
    plt.title('Цена акций AAPL и Скользящие Средние')
    plt.legend()

    # График RSI
    plt.subplot(4, 1, 2)
    plt.plot(df.index, df['rsi'], label='RSI', color='purple')
    plt.axhline(70, linestyle='--', alpha=0.5, color='red')
    plt.axhline(30, linestyle='--', alpha=0.5, color='green')
    plt.title('Индекс Относительной Силы (RSI)')
    plt.legend()

    # График MACD
    plt.subplot(4, 1, 3)
    plt.plot(df.index, df['macd'], label='MACD', color='blue')
    plt.plot(df.index, df['macd_signal'], label='Сигнальная линия MACD', color='orange')
    plt.title('MACD')
    plt.legend()

    # График стохастического осциллятора
    plt.subplot(4, 1, 4)
    plt.plot(df.index, df['stochastic'], label='Стохастический осциллятор', color='brown')
    plt.axhline(80, linestyle='--', alpha=0.5, color='red')
    plt.axhline(20, linestyle='--', alpha=0.5, color='green')
    plt.title('Стохастический осциллятор')
    plt.legend()

    plt.tight_layout()
    plt.show()