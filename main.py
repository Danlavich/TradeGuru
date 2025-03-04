from GetData import get_stock_data
from iluxa_finam_parser import finam_pars, visualisation
from Artemy_metrics import calculateMetrics
from NeuroForecast import predict

#данные с Yahoo
#df = get_stock_data("AAPL", "2024-01-01", "2024-02-01")

#данные с finam
df, name = finam_pars()
visualisation(df, name)

#расчет метрик и визуализация
calculateMetrics(df)

#используем модель для прогноза
predict(df)