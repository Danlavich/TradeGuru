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
import importlib
import Models
import Prediction
import Artemy_metrics
importlib.reload(Prediction)
importlib.reload(Models)
importlib.reload(Artemy_metrics)

# Импорт данных 
df=Models.get_stock_data('GAZP','2020-03-23','2025-03-22', "1d")
df_=df.copy()

# Метрики Артемия
# DF=Artemy_metrics.calculateMetrics(df_)


# Препроцесс
df=Models.preprocess(df,'close')
df=df.interpolate(method='linear')


# Определение будущих индексов
last_date = df.index[-1]  
future_index = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=30, freq='D')

# Определение лучшей регресии
name_regr=Models.best_regr(df,'close',0)

# Прогноз
def prediction(name):

   if name=='log-log':
       regr=Prediction.log_log_regression_forecast(df,'close',future_index)
   if name=='linear':
       regr=Prediction.linear_regression(df,'close',future_index)
   
       
   arima=Prediction.auto_arima_forecast(df,'close',future_index)
   prophet=Prediction.fb_prophet_forecast(df,'close',future_index)
   cat=Prediction.catboost(df,'close',future_index)
   xgb=Prediction.xgbboost(df,'close',future_index)


   final=0.3*arima+0.2*prophet+0.25*cat+0.15*xgb+0.1*regr
   print(final)
    
pred=prediction(name_regr)



