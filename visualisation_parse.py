import requests
import yfinance as yf
import mplfinance as mpf
from io import StringIO
import matplotlib.pyplot as plt
import pandas as pd

def get_stock_data(ticker: str, start_date: str, end_date: str, interval: str = "1d"):
    """
    Получает исторические данные акций с использованием yfinance,
    а в случае неудачи — использует Finam.

    Добавляет столбец 'Ticker' для последующей визуализации.

    :param ticker: Тикер акции (например, "AAPL" или "SBER")
    :param start_date: Дата начала (формат "YYYY-MM-DD")
    :param end_date: Дата окончания (формат "YYYY-MM-DD")
    :param interval: Интервал данных (например, "1m", "5m", "1h", "1d", "1wk", "1mo")
    :return: DataFrame с данными и столбцом 'Ticker'
    """
    try:
        print(f"⏳ Пробуем загрузить данные с Yahoo Finance для {ticker}...")
        stock = yf.Ticker(ticker)
        data = stock.history(start=start_date, end=end_date, interval=interval)
        if data is not None and not data.empty:
            df = data[['Open', 'Close', 'Volume']].reset_index()
            # При получении через yfinance имя столбца даты – "Datetime"
            df.rename(columns={"Datetime": "date"}, inplace=True)
            df["Ticker"] = ticker
            print(f"✅ Данные загружены с Yahoo Finance для {ticker}")
            return df
        else:
            print(f"⚠ Нет данных в Yahoo Finance для {ticker}. Пробуем Finam...")
            raise Exception("Данные отсутствуют или ошибка в yfinance.")
    except Exception as e:
        print(f"⚠ Ошибка при загрузке данных с Yahoo Finance: {e}. Пробуем Finam...")
        # Загрузка через Finam
        start_date_finam = start_date.replace("-", "")
        end_date_finam = end_date.replace("-", "")
        timeframe_map = {
            "1m": 1, "5m": 2, "10m": 3, "15m": 4, "30m": 5,
            "1h": 6, "1d": 7, "1wk": 8, "1mo": 9
        }
        timeframe = timeframe_map.get(interval, 7)  # По умолчанию — 1 день

        # Преобразуем даты для Finam
        df_day = int(start_date_finam[6:])
        mf = int(start_date_finam[4:6]) - 1
        yf_val = int(start_date_finam[:4])
        dt = int(end_date_finam[6:])
        mt = int(end_date_finam[4:6]) - 1
        yt = int(end_date_finam[:4])

        params = {
            'market': 1,
            'em': 3,
            'code': ticker,
            'apply': 0,
            'df': df_day, 'mf': mf, 'yf': yf_val,
            'dt': dt, 'mt': mt, 'yt': yt,
            'p': timeframe,
            'f': f"{ticker}_{start_date_finam}_{end_date_finam}",
            'e': '.csv',
            'cn': ticker,
            'dtf': 1,
            'tmf': 1,
            'MSOR': 1,
            'mstimever': 1,
            'sep': 3,
            'sep2': 1,
            'datf': 5,
            'at': 1
        }

        url = "https://export.finam.ru/" + params['f'] + params['e']
        response = requests.get(url, params=params)
        if response.status_code == 200 and response.text.strip():
            csv_data = StringIO(response.text)
            df_finam = pd.read_csv(csv_data, delimiter=';', encoding='cp1251')
            df_finam.columns = ["date", "time", "open", "high", "low", "close", "vol"]
            df_finam['date'] = pd.to_datetime(df_finam['date'], format='%Y%m%d')
            df_finam.set_index('date', inplace=True)
            df_finam.drop(columns=['time'], inplace=True)
            df_finam["Ticker"] = ticker
            print(f"✅ Данные загружены с Finam для {ticker}")
            # Вернем DataFrame, сбросив индекс для унификации структуры
            return df_finam.reset_index()
        else:
            print(f"❌ Ошибка загрузки данных с Finam для {ticker}")
            return None
def visualisation(df):
    """
    Визуализирует график цены закрытия акций.

    Функция автоматически определяет столбец с датой и выбирает нужный столбец цены.
    Тикер для заголовка берется из столбца 'Ticker' (первое значение).

    :param df: DataFrame с данными акций
    """
    # Определяем имя тикера
    ticker = df["Ticker"].iloc[
        0] if "Ticker" in df.columns else "Unknown"  # Определяем, какой столбец с датой использовать
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df.set_index("date", inplace=True)
    elif "Datetime" in df.columns:
        df["Datetime"] = pd.to_datetime(df["Datetime"], errors="coerce")
        df.set_index("Datetime", inplace=True)
    else:
        raise KeyError("Не найден столбец с датой в DataFrame")

    # Определяем, какой столбец с ценой закрытия использовать
    if "Close" in df.columns:
        close_col = "Close"
    elif "close" in df.columns:
        close_col = "close"
    else:
        raise KeyError("Не найден столбец с ценой закрытия (Close/close) в DataFrame")

    plt.figure(figsize=(12, 6))
    plt.plot(df[close_col], label='Close', color='b')
    plt.title(f'{ticker} Shares')
    plt.xlabel('Date')
    plt.ylabel('Price ($)')
    plt.grid()
    plt.legend()
    plt.show()

df = get_stock_data("SBER", "2024-01-01", "2024-02-01", interval="1h")
print(df)

if df is not None:
    print(df.head())
    visualisation(df)

