import yfinance as yf
import pandas as pd


def get_stock_data(ticker: str, start_date: str, end_date: str):
    """
    Получает исторические данные акций по тикеру.

    :param ticker: Тикер акции (например, "AAPL" для Apple)
    :param start_date: Дата начала в формате "YYYY-MM-DD"
    :param end_date: Дата окончания в формате "YYYY-MM-DD"
    :return: DataFrame с колонками [Date, Open, Close, Volume]
    """
    # Создаем объект Ticker для указанного тикера с помощью библиотеки yf (yfinance)
    stock = yf.Ticker(ticker)
    
    # Получаем исторические данные акций за указанный период
    data = stock.history(start=start_date, end=end_date)
    
    # Оставляем только нужные колонки: 'Open', 'Close' и 'Volume'
    # Сбрасываем индекс, чтобы дата стала обычной колонкой
    df = data[['Open', 'Close', 'Volume']].reset_index()
    
    # Возвращаем DataFrame с историческими данными акций
    return df

# Пример использования
df = get_stock_data("AAPL", "2024-01-01", "2024-02-01")
print(df)
