import requests
from requests.exceptions import RequestException
import logging

logger = logging.getLogger(__name__)

class HttpClient:
    def __init__(self, base_url="https://tradingeconomics.com/"):
        self.base_url = base_url
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                          'AppleWebKit/537.36 (KHTML, like Gecko) '
                          'Chrome/91.0.4472.124 Safari/537.36'
        })

    def get_page(self, endpoint):
        url = f"{self.base_url.rstrip('/')}/{endpoint}"
        try:
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            return response.text
        except RequestException as e:
            logger.error(f"Failed to fetch {url}: {str(e)}")
            return None

    def close(self):
        self.session.close()