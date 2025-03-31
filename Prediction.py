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

def smape(y_true, y_pred):
    return np.mean(2 * np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred))) * 100


    

def auto_arima_forecast(df, feat, future_index, graph=0):
    df = df.copy()
    
    model_auto = auto_arima(df[[feat]], seasonal=False,
                            trace=False, suppress_warnings=True, stepwise=True)

  
    n_periods = len(future_index)
    forecast, conf_int = model_auto.predict(n_periods=n_periods, return_conf_int=True)

    # Создаём DataFrame прогноза
    forecast_df = pd.DataFrame(forecast.values, index=future_index, columns=['Прогноз'])

    test=df[df.index>'2025']

    if graph != 0:
        plt.figure(figsize=(8, 4))
        plt.plot(test.index, test[feat], label="История", color='blue')
        plt.plot(forecast_df.index, forecast_df['Прогноз'], label="Прогноз", color='orange', linestyle='dashed')
        plt.title('Auto ARIMA Forecast')
        plt.legend()
        plt.grid()
        plt.show()

    return forecast_df.values
    

def fb_prophet_forecast(df, target, future_index, graph=0):
    df_new = df[[target]].copy()
    df_new.reset_index(inplace=True)
    df_new.rename(columns={target: 'y', df.index.name: 'ds'}, inplace=True)
    df_new['ds'] = pd.to_datetime(df_new['ds'])

   
    logging.getLogger("cmdstanpy").setLevel(logging.ERROR)

    
    param_grid = {
        'changepoint_prior_scale': [0.01, 0.1, 0.5],
        'seasonality_prior_scale': [0.1, 1.0, 10.0],
        'seasonality_mode': ['additive', 'multiplicative']
    }
    
    best_params = None
    best_mse = float("inf")

    for params in ParameterGrid(param_grid):
        model = Prophet(
            changepoint_prior_scale=params['changepoint_prior_scale'],
            seasonality_prior_scale=params['seasonality_prior_scale'],
            seasonality_mode=params['seasonality_mode'],
            yearly_seasonality=True
        )
       
        model.fit(df_new)
        forecast_temp = model.predict(df_new[['ds']])
        mse = ((df_new['y'] - forecast_temp['yhat'])**2).mean()
        
        if mse < best_mse:
            best_mse = mse
            best_params = params


   
    best_model = Prophet(
        changepoint_prior_scale=best_params['changepoint_prior_scale'],
        seasonality_prior_scale=best_params['seasonality_prior_scale'],
        seasonality_mode=best_params['seasonality_mode'],
        yearly_seasonality=True
    )
    best_model.fit(df_new)
    future = pd.DataFrame({'ds': pd.to_datetime(future_index)})
    forecast = best_model.predict(future)


    return forecast[['yhat']].values




def log_log_regression_forecast(df,target,fut_inx, graph=0):
    
    
    X = df.drop(target, axis=1)
    y = df[target]
    

    X_log = pd.DataFrame(np.log(X + 40),index=X.index,columns=X.columns)
    y_log = np.log(y + 40)


    model = LinearRegression()
    model.fit(X_log, y_log)

    
    fut_size = len(fut_inx)
    future_df = df.iloc[-30:].copy()
    fut_features = []
    
    for _ in range(fut_size):
        row = {}
            
        row['lag_1'] = future_df[target].iloc[-1]
        row['lag_2'] = future_df[target].iloc[-2]
        row['lag_3'] = future_df[target].iloc[-3]
        row['rolling_mean_3'] = np.mean([row['lag_1'], row['lag_2'], row['lag_3']])
    
        ext_series = pd.concat([future_df[target], pd.Series([row['lag_1']])], ignore_index=True)
        row['ema_5'] = ext_series.ewm(span=5, adjust=False).mean().iloc[-1]
        row['pct_change'] = (row['lag_1'] - row['lag_2']) / row['lag_2'] * 100 if row['lag_2'] != 0 else 0
        row['volatility_5'] = ext_series[-5:].std()
        
        delta = ext_series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(5).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(5).mean()
        rs = gain / loss
        row['rsi_5'] = 100 - (100 / (1 + rs)).iloc[-1]
    
        ema12 = ext_series.ewm(span=12, adjust=False).mean()
        ema26 = ext_series.ewm(span=26, adjust=False).mean()
        macd = ema12 - ema26
        row['macd'] = macd.iloc[-1]
        row['macd_signal'] = macd.ewm(span=9, adjust=False).mean().iloc[-1]
    
        ma = ext_series.rolling(20).mean()
        std = ext_series.rolling(20).std()
        row['bollinger_upper'] = ma.iloc[-1] + 2 * std.iloc[-1]
        row['bollinger_lower'] = ma.iloc[-1] - 2 * std.iloc[-1]
    
        max_p = ext_series[-30:].max()
        min_p = ext_series[-30:].min()
        diff = max_p - min_p
        row['fib_0.236'] = max_p - 0.236 * diff
        row['fib_0.382'] = max_p - 0.382 * diff
        row['fib_0.618'] = max_p - 0.618 * diff
    
        row['rsi_cluster'] = 1 if row['rsi_5'] > 70 else (0 if row['rsi_5'] < 30 else 0.5)
        row['vol_cluster'] = 1 if row['volatility_5'] > future_df[target].rolling(10).std().mean() else 0
    
        new_point = pd.DataFrame([row])
        new_point_log=pd.DataFrame(np.log(new_point+40))
        
        new_point[target] = model.predict(new_point_log)
        new_point[target] = np.exp(new_point[target]) - 40
    
        future_df = pd.concat([future_df, new_point], ignore_index=True)
        fut_features.append(new_point)


    future_forecast = pd.concat(fut_features, ignore_index=True)
    

    return  future_forecast[[target]].values



def linear_regression(df,target,fut_inx, graph=0):
    
   
    X = df.drop(target, axis=1)
    y = df[target]
    
   

    model = LinearRegression()
    model.fit(X, y)
    

    fut_size = len(fut_inx) 
    future_df = df.iloc[-30:].copy()
    fut_features = []
    
    for _ in range(fut_size):
        row = {}
    
        row['lag_1'] = future_df[target].iloc[-1]
        row['lag_2'] = future_df[target].iloc[-2]
        row['lag_3'] = future_df[target].iloc[-3]
        row['rolling_mean_3'] = np.mean([row['lag_1'], row['lag_2'], row['lag_3']])
    
        ext_series = pd.concat([future_df[target], pd.Series([row['lag_1']])], ignore_index=True)
        row['ema_5'] = ext_series.ewm(span=5, adjust=False).mean().iloc[-1]
        row['pct_change'] = (row['lag_1'] - row['lag_2']) / row['lag_2'] * 100 if row['lag_2'] != 0 else 0
        row['volatility_5'] = ext_series[-5:].std()
        
        delta = ext_series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(5).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(5).mean()
        rs = gain / loss
        row['rsi_5'] = 100 - (100 / (1 + rs)).iloc[-1]
    
        ema12 = ext_series.ewm(span=12, adjust=False).mean()
        ema26 = ext_series.ewm(span=26, adjust=False).mean()
        macd = ema12 - ema26
        row['macd'] = macd.iloc[-1]
        row['macd_signal'] = macd.ewm(span=9, adjust=False).mean().iloc[-1]
    
        ma = ext_series.rolling(20).mean()
        std = ext_series.rolling(20).std()
        row['bollinger_upper'] = ma.iloc[-1] + 2 * std.iloc[-1]
        row['bollinger_lower'] = ma.iloc[-1] - 2 * std.iloc[-1]
    
        max_p = ext_series[-30:].max()
        min_p = ext_series[-30:].min()
        diff = max_p - min_p
        row['fib_0.236'] = max_p - 0.236 * diff
        row['fib_0.382'] = max_p - 0.382 * diff
        row['fib_0.618'] = max_p - 0.618 * diff
    
        row['rsi_cluster'] = 1 if row['rsi_5'] > 70 else (0 if row['rsi_5'] < 30 else 0.5)
        row['vol_cluster'] = 1 if row['volatility_5'] > future_df[target].rolling(10).std().mean() else 0
    
        new_point = pd.DataFrame([row])
        new_point[target] = model.predict(new_point)
    
        future_df = pd.concat([future_df, new_point], ignore_index=True)
        fut_features.append(new_point)


    future_forecast = pd.concat(fut_features, ignore_index=True)
    
   
    return future_forecast[[target]].values






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
    model.fit(X_train, y_train, eval_set=(X_val, y_val), early_stopping_rounds=50, verbose=False)
    

    y_pred = model.predict(X_val)
    smape_val = smape(y_val, y_pred)

    return smape_val


def catboost(df,target,fut_inx, graph=1):
    fut_size = len(fut_inx) - 1
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

    model.fit(X,y,verbose=False)

    fut_size = len(fut_inx) 
    future_df = df.iloc[-30:].copy()
    fut_features = []
    
    for _ in range(fut_size):
        row = {}
    
        row['lag_1'] = future_df[target].iloc[-1]
        row['lag_2'] = future_df[target].iloc[-2]
        row['lag_3'] = future_df[target].iloc[-3]
        row['rolling_mean_3'] = np.mean([row['lag_1'], row['lag_2'], row['lag_3']])
    
        ext_series = pd.concat([future_df[target], pd.Series([row['lag_1']])], ignore_index=True)
        row['ema_5'] = ext_series.ewm(span=5, adjust=False).mean().iloc[-1]
        row['pct_change'] = (row['lag_1'] - row['lag_2']) / row['lag_2'] * 100 if row['lag_2'] != 0 else 0
        row['volatility_5'] = ext_series[-5:].std()
        
        delta = ext_series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(5).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(5).mean()
        rs = gain / loss
        row['rsi_5'] = 100 - (100 / (1 + rs)).iloc[-1]
    
        ema12 = ext_series.ewm(span=12, adjust=False).mean()
        ema26 = ext_series.ewm(span=26, adjust=False).mean()
        macd = ema12 - ema26
        row['macd'] = macd.iloc[-1]
        row['macd_signal'] = macd.ewm(span=9, adjust=False).mean().iloc[-1]
    
        ma = ext_series.rolling(20).mean()
        std = ext_series.rolling(20).std()
        row['bollinger_upper'] = ma.iloc[-1] + 2 * std.iloc[-1]
        row['bollinger_lower'] = ma.iloc[-1] - 2 * std.iloc[-1]
    
        max_p = ext_series[-30:].max()
        min_p = ext_series[-30:].min()
        diff = max_p - min_p
        row['fib_0.236'] = max_p - 0.236 * diff
        row['fib_0.382'] = max_p - 0.382 * diff
        row['fib_0.618'] = max_p - 0.618 * diff
    
        row['rsi_cluster'] = 1 if row['rsi_5'] > 70 else (0 if row['rsi_5'] < 30 else 0.5)
        row['vol_cluster'] = 1 if row['volatility_5'] > future_df[target].rolling(10).std().mean() else 0
    
        new_point = pd.DataFrame([row])
        new_point[target] = model.predict(new_point)
    
        future_df = pd.concat([future_df, new_point], ignore_index=True)
        fut_features.append(new_point)


    future_forecast = pd.concat(fut_features, ignore_index=True)

  

    return future_forecast[[target]].values




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
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    y_pred = model.predict(X_val)
    smape_val = smape(y_val, y_pred) 

    return smape_val  





def xgbboost(df,target,fut_inx, graph=1):
    
    
    X = df.drop(target, axis=1)
    y = df[target]

    
    train_size = int(len(df) * 0.7)
    val_size = int(len(df) * 0.1)

    X_train, X_val, X_test = X[:train_size], X[train_size:train_size+val_size], X[train_size+val_size:]
    y_train, y_val, y_test = y[:train_size], y[train_size:train_size+val_size], y[train_size+val_size:]

   
    study = optuna.create_study(direction="minimize")
    study.optimize(lambda trial: objective_xgb(trial, X_train, X_val, y_train, y_val), n_trials=30)

    best_params = study.best_params
    model = XGBRegressor(**best_params)


    model.fit(X,y,verbose=False)
    
    fut_size = len(fut_inx) 
    future_df = df.iloc[-30:].copy()
    fut_features = []
    
    for _ in range(fut_size):
        row = {}
    
        row['lag_1'] = future_df[target].iloc[-1]
        row['lag_2'] = future_df[target].iloc[-2]
        row['lag_3'] = future_df[target].iloc[-3]
        row['rolling_mean_3'] = np.mean([row['lag_1'], row['lag_2'], row['lag_3']])
    
        ext_series = pd.concat([future_df[target], pd.Series([row['lag_1']])], ignore_index=True)
        row['ema_5'] = ext_series.ewm(span=5, adjust=False).mean().iloc[-1]
        row['pct_change'] = (row['lag_1'] - row['lag_2']) / row['lag_2'] * 100 if row['lag_2'] != 0 else 0
        row['volatility_5'] = ext_series[-5:].std()
        
        delta = ext_series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(5).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(5).mean()
        rs = gain / loss
        row['rsi_5'] = 100 - (100 / (1 + rs)).iloc[-1]
    
        ema12 = ext_series.ewm(span=12, adjust=False).mean()
        ema26 = ext_series.ewm(span=26, adjust=False).mean()
        macd = ema12 - ema26
        row['macd'] = macd.iloc[-1]
        row['macd_signal'] = macd.ewm(span=9, adjust=False).mean().iloc[-1]
    
        ma = ext_series.rolling(20).mean()
        std = ext_series.rolling(20).std()
        row['bollinger_upper'] = ma.iloc[-1] + 2 * std.iloc[-1]
        row['bollinger_lower'] = ma.iloc[-1] - 2 * std.iloc[-1]
    
        max_p = ext_series[-30:].max()
        min_p = ext_series[-30:].min()
        diff = max_p - min_p
        row['fib_0.236'] = max_p - 0.236 * diff
        row['fib_0.382'] = max_p - 0.382 * diff
        row['fib_0.618'] = max_p - 0.618 * diff
    
        row['rsi_cluster'] = 1 if row['rsi_5'] > 70 else (0 if row['rsi_5'] < 30 else 0.5)
        row['vol_cluster'] = 1 if row['volatility_5'] > future_df[target].rolling(10).std().mean() else 0
    
        new_point = pd.DataFrame([row])
        new_point[target] = model.predict(new_point)
    
        future_df = pd.concat([future_df, new_point], ignore_index=True)
        fut_features.append(new_point)


    future_forecast = pd.concat(fut_features, ignore_index=True)
    
    
    return future_forecast[[target]].values
