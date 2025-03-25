import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Функция для получения исторических данных акций
# ticker: строка с символом акции, например "AAPL"
# start_date: строка с начальной датой в формате "YYYY-MM-DD"
# end_date: строка с конечной датой в формате "YYYY-MM-DD"
# Возвращает DataFrame с колонками 'Open', 'Close', 'Volume' и индексом по датам
def get_stock_data(ticker: str, start_date: str, end_date: str):
    stock = yf.Ticker(ticker)
    data = stock.history(start=start_date, end=end_date)
    df = data[['Open', 'Close', 'Volume']].reset_index()
    df.columns = ['Date', 'Open', 'Close', 'Volume']  # Set column names to uppercase
    return df

# Получение данных акций Apple с 1 января по 1 февраля 2024 года
df = get_stock_data("AAPL", "2024-01-01", "2024-02-01")
df.set_index('Date', inplace=True) # Установка даты в качестве индекса DataFrame
#df.rename(columns={'Close': 'close', 'Volume': 'volume'}, inplace=True) # Переименование колонок для удобства

# 1. Объём продаж (volume)
# Возвращает колонку 'volume' из DataFrame
def calculate_volume(df):
    return df['Volume']

# 2. Скользящие средние (Moving Average)
# window: целое число, задающее окно для скользящей средней
# Возвращает скользящую среднюю закрытия на заданном окне
def moving_average(df, window):
    return df['Close'].rolling(window=window).mean()

# 3. Экспоненциальные скользящие средние (Exponential Moving Average)
# window: целое число, задающее окно для экспоненциальной скользящей средней
# Возвращает экспоненциальную скользящую среднюю закрытия на заданном окне
def exponential_moving_average(df, window):
    return df['Close'].ewm(span=window, adjust=False).mean()

# 4. Индекс относительной силы (RSI)
# window: целое число, задающее окно для расчета RSI (по умолчанию 14)
# Рассчитывает RSI на основе изменений цен закрытия и возвращает его значение
def calculate_rsi(df, window=14):
    delta = df['Close'].diff() # Изменения цен закрытия
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean() # Средний прирост
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean() # Среднее снижение
    rs = gain / loss # Отношение прироста к снижению
    rsi = 100 - (100 / (1 + rs)) # Расчет RSI
    return rsi

# 5. Уровни Фибоначчи
# Рассчитывает уровни Фибоначчи на основе максимальной и минимальной цены закрытия
# Возвращает словарь с уровнями Фибоначчи и максимальной/минимальной ценами
def fibonacci_levels(df):
    max_price = df['Close'].max() # Максимальная цена
    min_price = df['Close'].min() # Минимальная цена
    diff = max_price - min_price # Разница между максимальной и минимальной ценами
    level1 = max_price - 0.236 * diff # Уровень 23.6%
    level2 = max_price - 0.382 * diff # Уровень 38.2%
    level3 = max_price - 0.618 * diff # Уровень 61.8%
    return {
        'level1': level1,
        'level2': level2,
        'level3': level3,
        'max_price': max_price,
        'min_price': min_price
    }

# 6. Стандартное отклонение (Volatility)
# window: целое число, задающее окно для расчета стандартного отклонения
# Рассчитывает стандартное отклонение цен закрытия на заданном окне и возвращает его значение
def calculate_volatility(df, window=20):
    return df['Close'].rolling(window=window).std()

# 7. Процент изменения (Percentage Change)
# Рассчитывает процентное изменение цен закрытия и возвращает его значение
def calculate_percentage_change(df):
    return df['Close'].pct_change() * 100

# 8. MACD
# Рассчитывает MACD (скользящая средняя конвергенции/дивергенции) на основе экспоненциальных скользящих средних
# Возвращает значение MACD
def calculate_macd(df):
    ema12 = exponential_moving_average(df, 12) # 12-дневная EMA
    ema26 = exponential_moving_average(df, 26) # 26-дневная EMA
    macd = ema12 - ema26 # Разница между EMA
    return macd

# 9. MACD Signal Line
# window: целое число, задающее окно для расчета сигнальной линии MACD (по умолчанию 9)
# Рассчитывает сигнальную линию MACD на основе значения MACD и возвращает ее
def calculate_macd_signal(df, window=9):
    macd = calculate_macd(df) # Значение MACD
    return macd.rolling(window=window).mean() # Сигнальная линия MACD

# 10. Bollinger Bands
# window: целое число, задающее окно для расчета полос Боллинджера (по умолчанию 20)
# num_std_dev: количество стандартных отклонений для расчета верхней и нижней полосы (по умолчанию 2)
# Рассчитывает верхнюю и нижнюю полосы Боллинджера и возвращает их значения
def calculate_bollinger_bands(df, window=20, num_std_dev=2):
    ma = moving_average(df, window) # Скользящая средняя
    rolling_std = df['Close'].rolling(window).std() # Стандартное отклонение
    upper_band = ma + (rolling_std * num_std_dev) # Верхняя полоса
    lower_band = ma - (rolling_std * num_std_dev) # Нижняя полоса
    return upper_band, lower_band

# Функция для расчета всех метрик
def calculateMetrics(df):
    #df.set_index('Date', inplace=True) # Установка даты в качестве индекса (не используется здесь)
    #df.rename(columns={'Close': 'close', 'Volume': 'volume'}, inplace=True) # Переименование колонок

    # Расчет различных метрик и добавление их в DataFrame
    df['Volume'] = calculate_volume(df) # Объем продаж
    df['moving_average_20'] = moving_average(df, window=20) # 20-дневная скользящая средняя
    df['exponential_moving_average_20'] = exponential_moving_average(df, window=20) # 20-дневная EMA
    df['rsi'] = calculate_rsi(df) # RSI
    df['volatility'] = calculate_volatility(df) # Волатильность
    df['percentage_change'] = calculate_percentage_change(df) # Процент изменения
    df['macd'] = calculate_macd(df) # MACD
    df['macd_signal'] = calculate_macd_signal(df) # Сигнальная линия MACD
    df['bollinger_upper'], df['bollinger_lower'] = calculate_bollinger_bands(df) # Полосы Боллинджера

    fib_levels = fibonacci_levels(df) # Уровни Фибоначчи

    # Вывод результатов
    print(df.tail()) # Печать последних строк DataFrame
    print("Уровни Фибоначчи:", fib_levels) # Печать уровней Фибоначчи

    # Визуализация
    plt.figure(figsize=(14, 10))

    # График цен и скользящих средних
    plt.subplot(3, 1, 1) # Создание подграфика
    plt.plot(df.index, df['Close'], label='Цена закрытия', color='blue') # График цены закрытия
    plt.plot(df.index, df['moving_average_20'], label='20-дневная скользящая средняя', color='orange') # График 20-дневной SMA
    plt.plot(df.index, df['exponential_moving_average_20'], label='20-дневная экспоненциальная скользящая средняя', color='green') # График 20-дневной EMA
    plt.title('Цена акций AAPL и Скользящие Средние') # Заголовок графика
    plt.legend() # Легенда графика

    # График RSI
    plt.subplot(3, 1, 2) # Создание подграфика
    plt.plot(df.index, df['rsi'], label='RSI', color='purple') # График RSI
    plt.axhline(70, linestyle='--', alpha=0.5, color='red') # Линия 70 на графике RSI
    plt.axhline(30, linestyle='--', alpha=0.5, color='green') # Линия 30 на графике RSI
    plt.title('Индекс Относительной Силы (RSI)') # Заголовок графика
    plt.legend() # Легенда графика

    # График MACD
    plt.subplot(3, 1, 3) # Создание подграфика
    plt.plot(df.index, df['macd'], label='MACD', color='blue') # График MACD
    plt.plot(df.index, df['macd_signal'], label='Сигнальная линия MACD', color='orange') # График сигнальной линии MACD
    plt.title('MACD') # Заголовок графика
    plt.legend() # Легенда графика

    plt.tight_layout() # Автоматическая подгонка подграфиков
    plt.show() # Отображение графиков

# Пример использования:
df = get_stock_data("AAPL", "2024-01-01", "2024-02-01")
df.set_index('Date', inplace=True)
calculateMetrics(df.copy())  # Pass a copy to avoid modifying the original DataFrame