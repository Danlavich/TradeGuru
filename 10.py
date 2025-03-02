import aiohttp
from DeeperSeek import DeepSeek, DeepSeekAPIError, InvalidCredentialsError

async def safe_api_call(api_method, *args, **kwargs):
    
    try:
        return await api_method(*args, **kwargs)

    except InvalidCredentialsError:
        print("[Ошибка аутентификации] Неверный логин или пароль.")
        return None

    except DeepSeekAPIError as e:
        print(f"[Ошибка API] {e}")
        return None

    except aiohttp.ClientError as e:
        print(f"[Ошибка сети] Проблема с подключением к API: {e}")
        return None

    except TimeoutError:
        print("[Ошибка сети] Время ожидания запроса истекло.")
        return None

    except Exception as e:
        print(f"[Неизвестная ошибка] {e}")
        return None
