import pandas as pd
import numpy as np
import scipy.stats as stats
from sklearn.metrics import root_mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error
import scipy.stats as stats
import statsmodels.api as sm
import logging
from prophet import Prophet


import matplotlib.pyplot as plt
from pmdarima import auto_arima
from statsmodels.tsa.arima.model import ARIMA
from math import ceil

import yfinance as yf
import pandas as pd
import requests
from io import StringIO
import matplotlib.pyplot as plt


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
        # timeframe_map = {
        #     "1m": 1, "5m": 2, "10m": 3, "15m": 4, "30m": 5,
        #     "1h": 6, "1d": 7, "1wk": 8, "1mo": 9
        # }
        timeframe_map = {'tick': 1, '1m': 2, '5m': 3, '10m': 4, '15m': 5, '30m': 6, '1h': 7, '1d': 8,
                         '1wk': 9, '1mo': 10}
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


df = get_stock_data("GAZP", "2024-01-01", "2024-04-02", interval="1d")

# if df is not None:
#     print(df.head())
#     visualisation(df)


def preprocess(df, target):
    # df=df.copy()
    df['date'] = pd.to_datetime(df['date'], format='%Y%m%d')
    df.set_index('date', inplace=True)
    df.index = pd.date_range(start=df.index[0], periods=len(df), freq='D')
    df.index.name = "date"
    df = df[[target]].copy()
    df['lag_1'] = df.iloc[:, 0].shift(1)
    df['lag_2'] = df.iloc[:, 0].shift(2)
    df['lag_3'] = df.iloc[:, 0].shift(3)

    df['rolling_mean_3'] = df.iloc[:, 0].rolling(window=3).mean()
    df['rolling_mean_6'] = df.iloc[:, 0].rolling(window=6).mean()

    df.dropna(inplace=True)
    return df


def auto_arima_prediction(Train, feat):
    if not isinstance(Train.index, pd.DatetimeIndex):
        Train.index = pd.to_datetime(Train.index)
    train_size = int(len(Train) * 0.8)
    train, test = Train[feat].iloc[:train_size], Train[feat].iloc[train_size:]

    model_auto = auto_arima(train, seasonal=False,
                            trace=False, suppress_warnings=True, stepwise=True)

    n_periods = len(test)
    forecast, conf_int = model_auto.predict(n_periods=n_periods, return_conf_int=True)

    forecast_index = test.index
    forecast_df = pd.DataFrame(forecast, index=forecast_index, columns=['Прогноз'])

    print(f'MSE auto_arima : {mean_squared_error(test, forecast)}')
    # return forecast_df['Прогноз'].values

    plt.figure(figsize=(10, 6))
    plt.plot(train.index, train, label="Train")
    plt.plot(test.index, test, label="Test", linestyle="dashed")
    plt.plot(test.index, forecast_df, label="Прогноз", linestyle="dotted", color="red")
    plt.title('ARIMA forecast')
    plt.legend()
    plt.show()


def fb_prophet_prediction(df, target):
    shape = df.shape[0]
    df_new = df[[target]].copy()
    df_new.reset_index(inplace=True)
    # df_new['Дата'] = pd.to_datetime(df_new['Дата'],format='%Y-%m-%d')
    # df_new.index = df_new['Дата']
    df_new.rename(columns={target: 'y', 'date': 'ds'}, inplace=True)
    train_set = df_new.iloc[:ceil(shape * 0.8)]
    valid_set = df_new.iloc[ceil(shape * 0.8):].copy()
    logging.getLogger("cmdstanpy").setLevel(logging.ERROR)

    model = Prophet()
    model.fit(train_set)
    close_prices = model.make_future_dataframe(periods=len(valid_set), freq='ME')
    forecast = model.predict(close_prices)
    forecast_valid = forecast['yhat'][ceil(shape * 0.8):]
    mse = mean_squared_error(valid_set['y'], forecast_valid)
    print('MSE prophet :', mse)
    valid_set.loc[:, 'Predictions'] = forecast_valid.values
    # return forecast_valid.values

    plt.figure(figsize=(10, 6))
    plt.plot(train_set['ds'], train_set['y'], color='red')
    plt.plot(valid_set['ds'], valid_set['y'], color='orange')
    plt.plot(valid_set['ds'], forecast_valid, color='b')
    plt.title('Прогноз  by FB Prophet')
    plt.show()

    # plt.plot(combined_actual.index, combined_actual.values, color='blue', label='Training Data')
    # plt.plot(valid_set.index, valid_set['y'], color='red', label='Validation Data')
    # plt.plot(valid_set['Predictions'],color='orange', linestyle='--', label='Predicted Data')
    # plt.xlabel('Год',size=20)
    # plt.ylabel('price')
    # plt.title('Прогноз  by FB Prophet')
    # plt.legend(['Model Training Data', 'Actual Data', 'Predicted Data'])
    # plt.show()


def power_set_recursive(s):
    if not s:
        return [[]]

    first = s[0]
    rest_subsets = power_set_recursive(s[1:])
    new_subsets = [[first] + subset for subset in rest_subsets]

    return rest_subsets + new_subsets


def log_log_regression(X, y):
    name = 'log-log'
    alpha = 0.05
    X_log = pd.DataFrame(np.log(X + 1), columns=X.columns, index=X.index)
    y_log = pd.Series(np.log(y + 1), index=y.index)

    X_log = sm.add_constant(X_log)

    train_size = int(len(X) * 0.8)
    X_train, X_test = X_log[:train_size], X_log[train_size:]
    y_train, y_test = y_log[:train_size], y_log[train_size:]

    model = sm.OLS(y_train, X_train).fit()
    p_values = model.pvalues
    y_pred = np.exp(model.predict(X_test))

    y_pred = np.exp(y_pred) - 1

    mse = mean_squared_error(y[train_size:], y_pred)

    if (p_values < alpha).all():
        return mse, name, y_pred
    else:
        return np.inf, np.inf, np.inf


def log_lin_regression(X, y):
    name = 'log-lin'
    alpha = 0.05
    X_transformed = pd.DataFrame(np.log(X + 1), columns=X.columns, index=X.index)
    y_transformed = y
    X_transformed = sm.add_constant(X_transformed)

    train_size = int(len(X) * 0.8)
    X_train, X_test = X_transformed[:train_size], X_transformed[train_size:]
    y_train, y_test = y_transformed[:train_size], y_transformed[train_size:]

    model = sm.OLS(y_train, X_train).fit()
    p_values = model.pvalues
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)

    if (p_values < alpha).all():
        return mse, name, y_pred
    else:
        return np.inf, np.inf, np.inf


def lin_log_regression(X, y):
    name = 'lin-log'
    alpha = 0.05
    X_transformed = X.copy()
    y_transformed = pd.Series(np.log(y + 1), index=y.index)
    # y_transformed = np.log(y + 1)
    X_transformed = sm.add_constant(X_transformed)

    train_size = int(len(X) * 0.8)
    X_train, X_test = X_transformed[:train_size], X_transformed[train_size:]
    y_train, y_test = y_transformed[:train_size], y_transformed[train_size:]

    model = sm.OLS(y_train, X_train).fit()
    p_values = model.pvalues

    y_pred = model.predict(X_test)
    y_pred = np.exp(y_pred) - 1
    # y_test= np.exp(y_test) - 1
    mse = mean_squared_error(y[train_size:], y_pred)
    # print(rmse)

    # plt.plot(y[:train_size],color='r',label='train')
    # plt.plot(y[train_size:],color='b',label='test')
    # plt.plot(y_pred,color='orange',label='pred')
    # plt.grid()
    # plt.legend()
    # plt.show()
    if (p_values < alpha).all():
        return mse, name, y_pred
    else:
        return np.inf, np.inf, np.inf


def polynomial_regression(X, y, degree=2):
    name = 'polynom'
    alpha = 0.05
    poly = PolynomialFeatures(degree=degree)
    X_poly = poly.fit_transform(X)
    X_poly = pd.DataFrame(X_poly, index=X.index)
    # X_poly=sm.add_constant(X_poly)

    train_size = int(len(X) * 0.8)
    X_train, X_test = X_poly[:train_size], X_poly[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    model = sm.OLS(y_train, X_train).fit()
    p_values = model.pvalues
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    if (p_values < alpha).all():
        return mse, name, y_pred
    else:
        return np.inf, np.inf, np.inf


def linear_regression(X, y):
    name = 'linear'
    alpha = 0.05
    X = sm.add_constant(X)

    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    model = sm.OLS(y_train, X_train).fit()
    p_values = model.pvalues
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    if (p_values < alpha).all():
        return mse, name, y_pred
    else:
        return np.inf, np.inf, np.inf


def best(df, target, features=None):
    X = df.drop(target, axis=1)
    y = df[target]

    train_size = int(len(X) * 0.8)
    best_mse = np.inf
    models_array = list()

    if features is None:
        features = power_set_recursive(list(X.columns))[1:]

    for feat in features:

        models_array.append(log_log_regression(X[feat], y))
        models_array.append(log_lin_regression(X[feat], y))
        models_array.append(lin_log_regression(X[feat], y))
        models_array.append(polynomial_regression(X[feat], y, degree=2))
        models_array.append(linear_regression(X[feat], y))
        curr_rmse = min(models_array, key=lambda x: x[0])
        if curr_rmse[0] != None:
            if curr_rmse[0] < best_mse:
                models_array.clear()
                best_mse = curr_rmse[0]
                best_model = curr_rmse
                best_feat = feat[0]
    print(f'best MSE {best_mse}')
    print(f'best Features {best_feat}')
    plt.figure(figsize=(10, 6))
    plt.plot(y, label='fact')
    plt.plot(y[train_size:].index, best_model[2], label='predict', color='r')
    plt.title(f'best model {best_model[1]}')
    plt.grid()
    plt.legend()
    plt.show()

def best_model(df, target='30-90'):
        auto_arima_prediction(df, target)
        fb_prophet_prediction(df, target)
        best(df, target, features=None)

    # print(f'best MSE Regression {best_mse}')
    # print(f'best Features  Regression {best_feat}')
    # print(f'best model  Regression {best_model[1]}')

df=preprocess(df,'close')

best_model(df,target='close')