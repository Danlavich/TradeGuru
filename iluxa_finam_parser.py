import pandas as pd
import requests
from io import StringIO
import matplotlib.pyplot as plt


def finam_pars():
    # ticker = input("Введите тикер компании: ").upper()
    # start_date = input("Введите дату начала (ГГГГ-ММ-ДД): ").replace("-", "")
    # end_date = input("Введите дату конца (ГГГГ-ММ-ДД): ").replace("-", "")
    # timeframe = int(input("Выберите интервал (1 - минута, 2 - 5 минут, 3 - 10 минут, 4 - 15 минут, 5 - 30 минут, 6 - час, 7 - день, 8 - неделя, 9 - месяц): "))

    ticker = "SBER"
    start_date = "20240101"
    end_date = "20240301"
    timeframe = 7

    df, mf, yf = int(start_date[6:]), int(start_date[4:6]) - 1, int(start_date[:4])
    dt, mt, yt = int(end_date[6:]), int(end_date[4:6]) - 1, int(end_date[:4])

    params = {
        'market': 1,
        'em': 3,
        'code': ticker,
        'apply': 0,
        'df': df, 'mf': mf, 'yf': yf,
        'dt': dt, 'mt': mt, 'yt': yt,
        'p': timeframe,
        'f': f"{ticker}_{start_date}_{end_date}",
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
        df = pd.read_csv(csv_data, delimiter=';', encoding='cp1251')

        # print("\n✅ Данные успешно загружены:")
        # print(df.head())
        return df, ticker

    else:
        print("❌ Ошибка загрузки данных с Финам. Проверьте параметры!")


def visualisation(df, name):
    df.columns = ["Date", "Time", "Open", "High", "Low", "Close", "Volume"]
    df['Date'] = pd.to_datetime(df['Date'], format='%Y%m%d')
    df.set_index('Date', inplace=True)
    df = df.drop('Time', axis=1)

    plt.figure(figsize=(12, 6))
    plt.plot(df['Close'], label='Close', color='b')
    plt.title(f'{name} shares ')
    plt.grid()
    plt.legend()
    plt.show()


def pusk():
    df, name = finam_pars()
    visualisation(df, name)
