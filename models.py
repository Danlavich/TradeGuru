import pandas as pd
import numpy as np
import scipy.stats as stats
from sklearn.metrics import root_mean_squared_error

import matplotlib.pyplot as plt
from pmdarima import auto_arima
from statsmodels.tsa.arima.model import ARIMA
from math import ceil
from prophet import Prophet
from statsmodels.tsa.api import VAR
import statsmodels.api as sm
from sklearn.preprocessing import PolynomialFeatures
import logging
from sklearn.preprocessing import StandardScaler
import optuna
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_percentage_error as MAPE 
from statsmodels.tsa.stattools import adfuller
from sklearn.model_selection import ParameterGrid
from sklearn.linear_model import LinearRegression
import optuna
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
import matplotlib.pyplot as plt
from pmdarima import auto_arima
from statsmodels.tsa.arima.model import ARIMA
from math import ceil
import yfinance as yf
import pandas as pd
import requests
from io import StringIO
import matplotlib.pyplot as plt
import importlib
import Models
import Prediction

def smape(y_true, y_pred):
    return np.mean(2 * np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred))) * 100



def preprocess(df, target):
    df=df.copy()
    try:
        df.drop(columns=['high','low'], inplace=True)
    except KeyError:
        pass
        
    df.columns=['date', 'open','close', 'vol', 'Ticker']
    
    df['date'] = pd.to_datetime(df['date'], format='%Y%m%d')
    df.set_index('date', inplace=True)
    df.index.name = "date" 
   
    df = df[[target]].copy()
    df['lag_1'] = df[target].shift(1)
    df['lag_2'] = df[target].shift(2)
    df['lag_3'] = df[target].shift(3)
    df['rolling_mean_3'] = df[['lag_1', 'lag_2', 'lag_3']].mean(axis=1)
    df['ema_5'] = df[target].ewm(span=5, adjust=False).mean()
    df['pct_change'] = df[target].pct_change() * 100
    df['volatility_5'] = df[target].rolling(5).std()
    
    delta = df[target].diff()
    gain = (delta.where(delta > 0, 0)).rolling(5).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(5).mean()
    rs = gain / loss
    df['rsi_5'] = 100 - (100 / (1 + rs))
    
    ema12 = df[target].ewm(span=12, adjust=False).mean()
    ema26 = df[target].ewm(span=26, adjust=False).mean()
    df['macd'] = ema12 - ema26
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    
    ma = df[target].rolling(20).mean()
    std = df[target].rolling(20).std()
    df['bollinger_upper'] = ma + 2 * std
    df['bollinger_lower'] = ma - 2 * std
    
    max_price = df[target].rolling(30).max()
    min_price = df[target].rolling(30).min()
    diff = max_price - min_price
    df['fib_0.236'] = max_price - 0.236 * diff
    df['fib_0.382'] = max_price - 0.382 * diff
    df['fib_0.618'] = max_price - 0.618 * diff
    
    df['rsi_cluster'] = df['rsi_5'].apply(lambda x: 1 if x > 70 else (0 if x < 30 else 0.5))
    rolling_vol = df[target].rolling(10).std().mean()
    df['vol_cluster'] = df['volatility_5'].apply(lambda x: 1 if x > rolling_vol else 0)
   
    df.dropna(inplace=True)
    return df.asfreq('D')

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
        timeframe_map={'tick': 1, '1m': 2, '5m': 3, '10m': 4, '15m': 5, '30m': 6, '1h': 7, '1d': 8, 
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



# if df is not None:
#     print(df.head())
#     visualisation(df)


def smape(y_true, y_pred):
    return np.mean(2 * np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred))) * 100
    

def auto_arima_prediction(df, feat, graph=0):
    df=df.copy()
    train_size = int(len(df) * 0.8)
    train, test = df.iloc[:train_size], df.iloc[train_size:]
    
    model_auto = auto_arima(train[feat], seasonal=False,
                            trace=False, suppress_warnings=True, stepwise=True)
    
    n_periods = len(test)
    forecast, conf_int = model_auto.predict(n_periods=n_periods, return_conf_int=True)
    
    forecast_index = test.index
    forecast_df = pd.DataFrame(forecast, index=forecast_index, columns=['Прогноз'])
    Smape = smape(test[feat], forecast)
    print(f'Smape Auto Arima {Smape}')
    
    if graph != 0:
        plt.figure(figsize=(8, 4))
        plt.plot(train.index, train[feat], label="Train", color='blue')
        plt.plot(test.index, test[feat], label="Test", color='red')
        plt.plot(forecast_df, label="Predicted", color="orange", linestyle='dashed')
        plt.title('ARIMA')
        plt.legend()
        plt.grid()
        plt.show()
    
    return Smape
    

def make_stationary(df):
    df = df.copy()
    last_values = df.iloc[-1]
    df_diff = df.diff().dropna()
    return df_diff, last_values

def restore_levels(diff_df, last_value):
    restored_df = diff_df.cumsum()
    restored_df = restored_df.add(last_value, axis='columns')
    return restored_df



def fb_prophet(df, target, graph=0):
    df_new = df[[target]].copy()
    df_new.reset_index(inplace=True)
    df_new.rename(columns={target: 'y', df.index.name: 'ds'}, inplace=True)
    df_new['ds'] = pd.to_datetime(df_new['ds'])
    # df_new = df_new.set_index('ds').asfreq('YS').reset_index()

    train_size = int(len(df_new) * 0.8)
    val_size = int(len(df_new) * 0.1)  # 10% данных на валидацию
    train_set = df_new.iloc[:train_size - val_size]  # 70% на обучение
    val_set = df_new.iloc[train_size - val_size:train_size]  # 10% на валидацию
    test_set = df_new.iloc[train_size:]  # 20% на тестирование

    logging.getLogger("cmdstanpy").setLevel(logging.ERROR)

    param_grid = {
        'changepoint_prior_scale': [0.01, 0.1, 0.5],
        'seasonality_prior_scale': [0.1, 1.0, 10.0],
        'seasonality_mode': ['additive','multiplicative']
    }
    grid = ParameterGrid(param_grid)
    best_params = None
    best_smape = float("inf")


   
    for params in grid:
        model = Prophet(
            changepoint_prior_scale=params['changepoint_prior_scale'],
            seasonality_prior_scale=params['seasonality_prior_scale'],
            seasonality_mode=params['seasonality_mode'],
            yearly_seasonality=True
        )
        model.fit(train_set)

        forecast = model.predict(val_set[['ds']])  
        current_smape = smape(val_set['y'].values, forecast['yhat'].values)
      

        if current_smape < best_smape:
            best_smape = current_smape
            best_params = params
  

    
    best_model = Prophet(
        changepoint_prior_scale=best_params['changepoint_prior_scale'],
        seasonality_prior_scale=best_params['seasonality_prior_scale'],
        seasonality_mode=best_params['seasonality_mode'],
        yearly_seasonality=True
    )
    best_model.fit(pd.concat([train_set, val_set]))  

    future = test_set[['ds']].copy() 
    forecast = best_model.predict(future)
    Smape = smape(test_set['y'].values, forecast['yhat'].values)

    print(f'Лучший SMAPE Prophet: {Smape}')  
    
    if graph != 0:
        plt.figure(figsize=(8, 4))
        plt.plot(train_set['ds'], train_set['y'], label='Train', color='blue')
        plt.plot(val_set['ds'], val_set['y'], label='Validation', color='green')
        plt.plot(test_set['ds'], test_set['y'], label='Test', color='red')
        plt.plot(test_set['ds'], forecast['yhat'], label='Predicted', color='orange', linestyle='dashed')
        plt.title('Prophet Model')
        plt.grid()
        plt.legend()
        plt.show()

    return Smape



def log_log_regression(X, y):
    name = 'log-log'
    alpha = 0.05
    X_log = pd.DataFrame(np.log(X + 40), columns=X.columns, index=X.index)
    y_log = pd.Series(np.log(y + 40), index=y.index)

    # X_log = sm.add_constant(X_log)

    train_size = int(len(X) * 0.8)
    X_train, X_test = X_log[:train_size], X_log[train_size:]
    y_train, y_test = y_log[:train_size], y_log[train_size:]

    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)

    X_train_scaled = sm.add_constant(X_train_scaled)
    X_test_scaled = sm.add_constant(X_test_scaled)

    model = sm.OLS(y_train, X_train_scaled).fit()
    p_values = model.pvalues
    y_pred = model.predict(X_test_scaled)
    y_pred = np.exp(y_pred) - 40

    Smape = smape (y[train_size:], y_pred)
    return Smape, name, y_pred


def linear_regression(X, y):
    name = 'linear'
    alpha = 0.05
    X = sm.add_constant(X)

    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)

    X_train_scaled = sm.add_constant(X_train_scaled)
    X_test_scaled = sm.add_constant(X_test_scaled)

    model = sm.OLS(y_train, X_train_scaled).fit()
    p_values = model.pvalues
    y_pred = model.predict(X_test_scaled)

    Smape = smape (y_test, y_pred)
    return Smape, name, y_pred


def best_regr(df, target, graph=0,features=None):
    X = df.drop(target, axis=1)
    y = df[target]

    train_size = int(len(X) * 0.8)
    best_smape = np.inf
    models_array = list()

    
    features=[X.columns]

    for feat in features:

        models_array.append(log_log_regression(X[feat], y))
        models_array.append(polynomial_regression(X[feat], y, degree=1))
        models_array.append(linear_regression(X[feat], y))
        curr_smape = min(models_array, key=lambda x: x[0])
        if curr_smape[0] != None:
            if curr_smape[0] < best_smape:
                models_array.clear()
                best_smape = curr_smape[0]
                best_model = curr_smape
                best_feat = feat

    
    if graph!=0:
        plt.figure(figsize=(8, 4))
        plt.plot(y[:train_size], label='Train', color='b')
        plt.plot(y[train_size:], label='Test', color='r')
        plt.plot(y[train_size:].index, best_model[2], label='Pred', color='orange')
        plt.title(f'best model {best_model[1]}')
        plt.grid()
        plt.legend()
        plt.show()
    return best_smape,best_model[1]


def objective_cat(trial, X_train, X_val, y_train, y_val):
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    params = {
        "iterations": trial.suggest_int("iterations", 100, 1000, step=100),
        "depth": trial.suggest_int("depth", 4, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1, 10),
        "random_strength": trial.suggest_float("random_strength", 1, 10),
        "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 1.0),
        "border_count": trial.suggest_int("border_count", 32, 255),
        "grow_policy": trial.suggest_categorical("grow_policy", ["SymmetricTree", "Depthwise", "Lossguide"]),
        "verbose": 0
    }

    model = CatBoostRegressor(**params)
    model.fit(X_train, y_train, eval_set=(X_val, y_val), early_stopping_rounds=50, verbose=0)
    
    
    y_pred = model.predict(X_val)
    smape_val = smape(y_val, y_pred)

    return smape_val


def catboost(df, target, graph=0):

    X = df.drop(target, axis=1)
    y = df[target]

    
    train_size = int(len(df) * 0.7)
    val_size = int(len(df) * 0.1)
    X_train, X_val, X_test = X[:train_size], X[train_size:train_size+val_size], X[train_size+val_size:]
    y_train, y_val, y_test = y[:train_size], y[train_size:train_size+val_size], y[train_size+val_size:]

    
    study = optuna.create_study(direction="minimize")
    study.optimize(lambda trial: objective_cat(trial, X_train, X_val, y_train, y_val), n_trials=30)

    best_params = study.best_params
    model = CatBoostRegressor(**best_params)

    
    model.fit(pd.concat([X_train, X_val]), pd.concat([y_train, y_val]), eval_set=(X_val, y_val), early_stopping_rounds=50, verbose=0)


    y_pred = model.predict(X_test)
    Smape = smape(y_test, y_pred)

    print(f'SMAPE CatBoost: {Smape}')
    
    if graph != 0:
        plt.figure(figsize=(8, 4))
        plt.plot(y[:train_size], label='Train', color='b')
        plt.plot(y[train_size:train_size+val_size], label='Validation', color='g')
        plt.plot(y[train_size+val_size:], label='Test', color='r')
        plt.plot(y[train_size+val_size:].index, y_pred, label='Pred', color='orange')
        plt.title('CatBoost')
        plt.grid()
        plt.legend()
        plt.show()

    return Smape



def objective_xgb(trial, X_train, X_val, y_train, y_val):
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000, step=100),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 1, 10),
        "reg_alpha": trial.suggest_float("reg_alpha", 0, 10),
        "grow_policy": trial.suggest_categorical("grow_policy", ["depthwise", "lossguide"])
    }

    model = XGBRegressor(**params, early_stopping_rounds=50)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=0)

    y_pred = model.predict(X_val)
    smape_val = smape(y_val, y_pred) 

    return smape_val  


def xgbboost(df, target, graph=0):

    X = df.drop(target, axis=1)
    y = df[target]

    
    train_size = int(len(df) * 0.7)
    val_size = int(len(df) * 0.1)

    X_train, X_val, X_test = X[:train_size], X[train_size:train_size+val_size], X[train_size+val_size:]
    y_train, y_val, y_test = y[:train_size], y[train_size:train_size+val_size], y[train_size+val_size:]

   
    study = optuna.create_study(direction="minimize")
    study.optimize(lambda trial: objective_xgb(trial, X_train, X_val, y_train, y_val), n_trials=30)

    best_params = study.best_params
    model = XGBRegressor(**best_params, early_stopping_rounds=50)

   
    model.fit(pd.concat([X_train, X_val]), pd.concat([y_train, y_val]), 
              eval_set=[(X_val, y_val)], verbose=0)

   
    y_pred = model.predict(X_test)
    Smape = smape(y_test, y_pred)

    print(f'SMAPE XGBoost: {Smape}')
    
    if graph != 0:
        plt.figure(figsize=(8, 4))
        plt.plot(y[:train_size], label='Train', color='b')
        plt.plot(y[train_size:train_size+val_size], label='Validation', color='g')
        plt.plot(y[train_size+val_size:], label='Test', color='r')
        plt.plot(y[train_size+val_size:].index, y_pred, label='Pred', color='orange')
        plt.title('XGBoost')
        plt.grid()
        plt.legend()
        plt.show()

    return Smape


