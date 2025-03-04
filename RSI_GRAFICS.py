def perform_technical_analysis(df):
    """
    Выполняет технический анализ данных:
    - Вычисляет RSI
    - Вычисляет свечной паттерн Morning Star
    - Строит график с индикаторами

    :param df: DataFrame с ценами акций
    """
    if df is None or df.empty:
        print("⚠ Нет данных для анализа!")
        return

    print("📊 Выполняем технический анализ...")

    # Рассчитываем RSI
    rsi_func = abstract.Function("RSI")
    df['rsi'] = rsi_func(df)

    # Рассчитываем свечной паттерн Morning Star
    morning_star = abstract.Function("CDLMORNINGSTAR")
    df['morning_star'] = morning_star(df)

    print("Последние 10 значений результата паттерна Morning Star:")
    print(df['morning_star'].tail())

    # График RSI
    rsi_plot = mpf.make_addplot(df['rsi'], panel=1, color='blue', ylabel='RSI')

    # График свечного анализа
    mpf.plot(df, type='candle', style='yahoo', addplot=rsi_plot, volume=True)