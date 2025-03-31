import flet as ft

#import parse (а кокнретно get_stock_data и create_graph_image)
#import metrics (calculateMetrics)


def main_view(page: ft.Page):
    text_field_style = {
        "label_style": ft.TextStyle(size=14, color=ft.Colors.BLUE_GREY_700),
        "border_color": ft.Colors.BLUE_GREY_200,
        "focused_border_color": ft.Colors.BLUE_ACCENT,
    }

    ticker_input = ft.TextField(label="Введите тикер", **text_field_style)
    start_date_input = ft.TextField(label="Дата начала (YYYY-MM-DD)", **text_field_style)
    end_date_input = ft.TextField(label="Дата окончания (YYYY-MM-DD)", **text_field_style)

    interval_input = ft.Dropdown(
        label="Интервал",
        options=[ft.dropdown.Option(o) for o in ["1d", "1h", "30m", "15m", "5m", "1m"]],
        **{k: v for k, v in text_field_style.items() if k != "focused_border_color"}
    )

    def analyze_data(e):
        ticker = ticker_input.value.strip()
        start_date = start_date_input.value.strip()
        end_date = end_date_input.value.strip()
        interval = interval_input.value

        df = get_stock_data(ticker, start_date, end_date, interval)
        if df is not None:
            image_ticker = create_graph_image(df)
            image_ta = calculateMetrics(df)
            analysis_result = "1"  # Замените на реальные данные
            #image_prognose=prediction тут будет функция получения графика прогноза Ильи

            page.clean()
            page.add(analysis_view(page, analysis_result, image_ticker, image_ta))
            page.update()
        else:
            print("Ошибка при получении данных.")

    analyze_button = ft.ElevatedButton(
        text="Analyze",
        icon=ft.icons.ANALYTICS,
        style=ft.ButtonStyle(
            padding=20,
            color=ft.Colors.WHITE,
            bgcolor=ft.Colors.BLUE_ACCENT,
        ),
        on_click=analyze_data
    )

    input_form = ft.Container(
        content=ft.Column(
            [
                ft.Text("Stock Data Helper", style="headlineMedium"),
                ticker_input,
                start_date_input,
                end_date_input,
                interval_input,
                analyze_button
            ],
            spacing=25,
            horizontal_alignment=ft.CrossAxisAlignment.CENTER
        ),
        padding=40,
        width=600,
        bgcolor=ft.Colors.WHITE,
        border_radius=15,
        shadow=ft.BoxShadow(
            spread_radius=1,
            blur_radius=25,
            color=ft.Colors.BLUE_100
        )
    )

    page.clean()
    page.add(
        ft.ListView(
            expand=True,
            auto_scroll=True,
            controls=[
                ft.Container(
                    content=input_form,
                    alignment=ft.alignment.center,
                    gradient=ft.LinearGradient(
                        begin=ft.alignment.top_center,
                        end=ft.alignment.bottom_center,
                        colors=[ft.Colors.BLUE_50, ft.Colors.WHITE]
                    ),
                    padding=40
                )
            ]
        )
    )
    page.update()

def analysis_view(page: ft.Page, result: str, image_data: str, image_ta, image_prognose):
    header_style = ft.TextStyle(size=24, weight=ft.FontWeight.BOLD, color=ft.Colors.BLUE_GREY_800)
    shadow = ft.BoxShadow(spread_radius=1, blur_radius=15, color=ft.Colors.BLUE_100)

    charts = ft.Column(
        height=700,
        scroll=ft.ScrollMode.ALWAYS,
        spacing=40,
        controls=[
            ft.Container(
                content=ft.Column([
                    ft.Text("График цен", style=header_style),
                    ft.Image(
                        src_base64=image_data,
                        width=800,
                        height=400,
                        fit=ft.ImageFit.CONTAIN
                    )
                ], spacing=15),
                padding=20,
                bgcolor=ft.Colors.WHITE,
                border_radius=15,
                shadow=shadow
            ),

            ft.Container(
                content=ft.Column([
                    ft.Text("Технический анализ", style=header_style),
                    ft.Image(
                        src_base64=image_ta,
                        width=800,
                        height=600,
                        fit=ft.ImageFit.CONTAIN
                    )
                ], spacing=15),
                padding=20,
                bgcolor=ft.Colors.WHITE,
                border_radius=15,
                shadow=shadow
            ),
            ft.Container(
                content=ft.Column([
                    ft.Text("прогнозный график", style=header_style),
                    ft.Image(
                        src_base64=image_prognose,
                        width=800,
                        height=400,
                        fit=ft.ImageFit.CONTAIN
                    )
                ], spacing=15),
                padding=20,
                bgcolor=ft.Colors.WHITE,
                border_radius=15,
                shadow=shadow
            ),
        ]
    )

    info_panel = ft.Column(
        spacing=30,
        controls=[
            ft.Container(
                content=ft.Column([
                    ft.Text("Дополнительная информация от языковой модели", style=header_style),
                    ft.TextField(
                        multiline=True,
                        read_only=True,
                        border_color=ft.Colors.TRANSPARENT,
                        height=200
                    )
                ]),
                padding=20,
                bgcolor=ft.Colors.WHITE,
                border_radius=15,
                shadow=shadow
            )
        ]
    )

    layout = ft.Container(
        content=ft.ListView(
            expand=True,
            auto_scroll=True,
            controls=[
                ft.Row(
                    [charts, info_panel],
                    spacing=40,
                    vertical_alignment=ft.CrossAxisAlignment.START
                )
            ]
        ),
        padding=40,
        expand=True,
        gradient=ft.LinearGradient(
            colors=[ft.Colors.BLUE_50, ft.Colors.WHITE]
        )
    )

    back_button = ft.ElevatedButton(
        text="Назад",
        icon=ft.Icons.ARROW_BACK,
        style=ft.ButtonStyle(
            padding=20,
            color=ft.Colors.BLUE_ACCENT
        ),
        on_click=lambda e: main_view(page)
    )

    return ft.ListView(
        expand=True,
        controls=[
            ft.Column(
                [layout, ft.Divider(height=10), back_button],
                spacing=30
            )
        ]
    )

def main(page: ft.Page):
    page.title = "Stock Data Helper"
    page.theme = ft.Theme(
        color_scheme=ft.ColorScheme(
            primary=ft.Colors.BLUE,
            secondary=ft.Colors.BLUE_ACCENT
        ),
        text_theme=ft.TextTheme(
            body_medium=ft.TextStyle(size=16),
            title_medium=ft.TextStyle(size=20, weight=ft.FontWeight.BOLD)
        )
    )
    page.fonts = {
        "Roboto": "https://fonts.googleapis.com/css2?family=Roboto:wght@400;500;700&display=swap"
    }
    main_view(page)

ft.app(target=main)