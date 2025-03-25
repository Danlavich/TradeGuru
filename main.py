from visualisation_parse import get_stock_data, visualisation
# from iluxa_finam_parser import finam_pars, visualisation
from Artemy_metrics import calculateMetrics
from NeuroForecast import predict_stock_price

#данные с Yahoo
#df = get_stock_data("AAPL", "2024-01-01", "2024-02-01")


ticker = "SBER"
start_date = "2024-01-01"
end_date = "2024-02-01"
interval = "1d"

df = get_stock_data(ticker, start_date, end_date, interval)
visualisation(df)

#расчет метрик и визуализация
calculateMetrics(df)

#используем модель для прогноза
predict_stock_price(df)
