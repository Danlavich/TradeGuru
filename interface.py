import flet as ft
from visualisation_parse import get_stock_data, visualisation
def interface(page: ft.Page):
    page.title = "Stock Data Visualizer"

    ticker_input = ft.TextField(label="Введите тикер")
    start_date_input = ft.TextField(label="Дата начала (YYYY-MM-DD)")
    end_date_input = ft.TextField(label="Дата окончания (YYYY-MM-DD)")

    interval_input = ft.Dropdown(
        label="Интервал",
        options=[
            ft.dropdown.Option("1d"),
            ft.dropdown.Option("1h"),
            ft.dropdown.Option("30m"),
            ft.dropdown.Option("15m"),
            ft.dropdown.Option("5m"),
            ft.dropdown.Option("1m")
        ]
    )

    def analyze_data(e):
        ticker = ticker_input.value
        start_date = start_date_input.value
        end_date = end_date_input.value
        interval = interval_input.value

        df = get_stock_data(ticker, start_date, end_date, interval)
        if df is not None:
            print(df.head())
            visualisation(df)
        # Функции Артемия Дани Матвея Илюхи

    analyze_button = ft.ElevatedButton(text="Analyze", on_click=analyze_data)

    page.add(
        ft.Column([
            ticker_input,
            start_date_input,
            end_date_input,
            interval_input,
            analyze_button
        ])
    )

ft.app(target=interface)
