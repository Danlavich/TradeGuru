import requests
from bs4 import BeautifulSoup


def get_fundamental_data(ticker: str, country: str, metric: str):
    """
    Функция получает значение финансового показателя для компании.
    :param ticker: Тикер компании (например, AAPL)
    :param country: Код страны (например, US)
    :param metric: Финансовый показатель (например, EBITDA)
    :return: Значение показателя или None, если не найдено
    """
    url = f"https://tradingeconomics.com/{ticker.lower()}:{country.lower()}:{metric.lower()}"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/89.0.4389.82 Safari/537.36"}

    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        print(f"Ошибка при получении данных: {response.status_code}")
        return None

    soup = BeautifulSoup(response.text, "html.parser")

    # Поиск заголовка с нужным показателем
    card_body = soup.find("div", class_="card-body")
    if card_body:
        h2_text = card_body.find("h2")
        if h2_text:
            return h2_text.text.strip()

    print("Не удалось найти данные на странице.")
    return None


if __name__ == "__main__":
    ticker = input("Введите тикер компании (например, AAPL): ").strip().upper()
    country = input("Введите код страны (например, US): ").strip().upper()
    metric = input("Введите финансовый показатель (например, EBITDA): ").strip().upper()

    value = get_fundamental_data(ticker, country, metric)
    if value:
        print(f"{ticker}:{country} {metric}: {value}")
