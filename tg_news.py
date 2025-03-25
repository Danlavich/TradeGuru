import asyncio
from telethon import TelegramClient
from datetime import datetime, timezone
from dateutil import parser

async def fetch_telegram_news(channel_username, api_id, api_hash, phone, start_date, end_date):
    async with TelegramClient("session_name", api_id, api_hash) as client:
        await client.connect()

        if not await client.is_user_authorized():
            await client.send_code_request(phone)
            code = input("Введите код из Telegram: ")
            await client.sign_in(phone, code)

        messages = []

        print(f"📡 Загружаем сообщения с {start_date} по {end_date}...")

 
        async for message in client.iter_messages(channel_username, offset_date=end_date, reverse=False):
            if message.date < start_date:
                break  
            if start_date <= message.date <= end_date:
                if message.text:
                    messages.append(f"{message.date}: {message.text}\n\n")
                else:
                    messages.append(f"{message.date}: [Нет текста, возможно медиа или ссылка]\n\n")

      
        filename = f"{channel_username}_{start_date.date()}_to_{end_date.date()}.txt"
        with open(filename, "w", encoding="utf-8") as file:
            file.writelines(messages)

        print(f"\n✅ Сохранено {len(messages)} сообщений в файл: {filename}")


def get_news_by_date(channel_username, start_str, end_str):
    api_id = 28604669
    api_hash = "c5b7c5b54aceb2eb7f9424ef614b54c2"
    phone = "+79373711555"


    start_date = parser.parse(start_str).replace(tzinfo=timezone.utc)
    end_date = parser.parse(end_str).replace(tzinfo=timezone.utc)

    asyncio.run(fetch_telegram_news(channel_username, api_id, api_hash, phone, start_date, end_date))

if __name__ == "__main__":
    get_news_by_date("cb_economics", "2025-01-10", "2025-01-15")
