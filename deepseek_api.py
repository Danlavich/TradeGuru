import asyncio
from DeeperSeek import DeepSeek

async def start_deepseek():
    api = DeepSeek(
        email="email",  # Замените на ваш email
        password="password",  # Замените на ваш пароль
        token="token",  # Опционально, если используете токен
        chat_id="YOUR_CHAT_ID",  # Опционально
        chrome_args=[],
        verbose=True,  # Включите для отладки
        headless=False,  # Отключите headless для визуальной отладки
        attempt_cf_bypass=False,
    )
    await api.initialize()  # Инициализация API
    print("DeepSeek session is initialized")
    return api

async def message_deepseek(api):
    response = await api.send_message(
        "Hey DeepSeek!",
        deepthink=True,  # Использовать DeepThink
        search=False,  # Использовать поиск
        slow_mode=True,  # Отправлять сообщение в медленном режиме
        slow_mode_delay=0.25,  # Задержка между символами
        timeout=180,  # Время ожидания ответа
    )
    print("Response:", response.text)

async def main():
    api = await start_deepseek()  # Инициализация API
    #await message_deepseek(api)  # Отправка сообщения

if __name__ == "__main__":
    asyncio.run(main())