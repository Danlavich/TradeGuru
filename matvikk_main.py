import logging
from matvikk_http_client import HttpClient
from matvikk_parser import TradingEconomicsParser
from typing import Dict, Optional
import time

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


class FundamentalDataScraper:
    def __init__(self):
        self.http_client = HttpClient()
        self.parser = TradingEconomicsParser()

    def get_metric_data(
            self,
            ticker: str,
            country: str,
            metric: str
    ) -> Optional[str]:
        endpoint = f"{ticker.lower()}:{country.lower()}:{metric}"
        html_content = self.http_client.get_page(endpoint)
        return self.parser.parse_fundamental_data(html_content)

    def get_all_metrics_data(
            self,
            ticker: str,
            country: str
    ) -> Dict[str, Optional[str]]:
        metrics = self.parser.get_all_metrics()
        results = {}

        for metric_key, metric_name in metrics.items():
            data = self.get_metric_data(ticker, country, metric_key)
            results[metric_name] = data
            time.sleep(1)  # Чтобы избежать блокировки

        return results

    def close(self):
        self.http_client.close()


def format_output(
        ticker: str,
        country: str,
        results: Dict[str, Optional[str]]
) -> None:
    company_name = ticker.upper()
    print(f"\nFundamental data for {company_name} ({country.upper()}):\n")

    for metric_name, data in results.items():
        if data:
            print(data)
        else:
            print(
                f"No data available for {metric_name} "
                f"of {company_name} ({country.upper()})"
            )


def main():
    try:
        scraper = FundamentalDataScraper()

        ticker = input("Enter company ticker (e.g., AAPL): ").strip()
        country = input("Enter country code (e.g., US): ").strip()

        if not ticker or not country:
            print("Error: Ticker and country code are required.")
            return

        print("\nFetching data... This may take a few moments.\n")

        results = scraper.get_all_metrics_data(ticker, country)
        format_output(ticker, country, results)

    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")
    finally:
        scraper.close()


if __name__ == "__main__":
    main()