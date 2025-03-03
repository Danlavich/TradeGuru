
import asyncio
from telethon import TelegramClient

async def fetch_telegram_news(channel_username, api_id, api_hash, phone):
    # 🔹 Инициализируем клиента Telethon внутри функции
    async with TelegramClient("session_name", api_id, api_hash) as client:
        await client.connect()

        # 🔹 Проверяем авторизацию
        if not await client.is_user_authorized():
            await client.send_code_request(phone)
            code = input("Введите код из Telegram: ")
            await client.sign_in(phone, code)

        messages = []

        # 🔹 Загружаем последние 100 сообщений
        async for message in client.iter_messages(channel_username, limit=100):
            if message.text:
                messages.append(f"{message.date}: {message.text}\n")
            else:
                messages.append(f"{message.date}: [Нет текста, возможно ссылка или медиа]\n")

        # 🔹 Сохраняем в TXT-файл
        filename = f"{channel_username}_news.txt"
        with open(filename, "w", encoding="utf-8") as file:
            file.writelines(messages)

        print(f"\n✅ Новости сохранены в файл '{filename}'")


def get_news(channel_username):
    api_id = 28604669
    api_hash = "c5b7c5b54aceb2eb7f9424ef614b54c2"
    phone = "+79373711555"

    try:
        loop = asyncio.get_running_loop()
        task = loop.create_task(fetch_telegram_news(channel_username, api_id, api_hash, phone))
    except RuntimeError:
        asyncio.run(fetch_telegram_news(channel_username, api_id, api_hash, phone))


get_news("alfawealth")