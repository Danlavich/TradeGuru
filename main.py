from visualisation_parse import get_stock_data, visualisation
from Artemy_metrics import calculateMetrics
from NeuroForecast import predict_stock_price
from tg_news import get_news_by_date

#читаем тг новости
# get_news_by_date("cb_economics", "2025-01-01", "2025-01-10")

#входные данные для парсера
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
