from bs4 import BeautifulSoup
import logging
from typing import Optional, Dict

logger = logging.getLogger(__name__)


class TradingEconomicsParser:
    @staticmethod
    def parse_fundamental_data(html_content: str) -> Optional[str]:
        if not html_content:
            return None

        try:
            soup = BeautifulSoup(html_content, 'lxml')
            description_div = soup.find(
                'div',
                class_='card-body',
                style="text-align: justify"
            )

            if description_div:
                h2_tag = description_div.find('h2')
                if h2_tag:
                    return h2_tag.get_text(strip=True)
            return None
        except Exception as e:
            logger.error(f"Error parsing HTML: {str(e)}")
            return None

    @staticmethod
    def get_all_metrics() -> Dict[str, str]:
        return {
            'stock-price': 'Stock Price',
            'assets': 'Assets',
            'cash-and-equivalent': 'Cash and Equivalent',
            'cost-of-sales': 'Cost of Sales',
            'current-assets': 'Current Assets',
            'current-liabilities': 'Current Liabilities',
            'debt': 'Debt',
            'dividend-yield': 'Dividend Yield',
            'ebit': 'EBIT',
            'ebitda': 'EBITDA',
            'employees': 'Employees',
            'eps-earnings-per-share': 'EPS Earnings Per Share',
            'equity-capital-and-reserves': 'Equity Capital and Reserves',
            'gross-profit-on-sales': 'Gross Profit on Sales',
            'interest-expense-on-debt': 'Interest Expense on Debt',
            'interest-income': 'Interest Income',
            'loan-capital': 'Loan Capital',
            'market-capitalization': 'Market Capitalization',
            'net-income': 'Net Income',
            'operating-expenses': 'Operating Expenses',
            'operating-profit': 'Operating Profit',
            'ordinary-share-capital': 'Ordinary Share Capital',
            'pe-price-to-earnings': 'PE Price to Earnings',
            'pre-tax-profit': 'Pre-Tax Profit',
            'sales-revenues': 'Sales Revenues',
            'selling-and-administration-expenses': (
                'Selling and Administration Expenses'
            ),
            'stock': 'Stock',
            'trade-creditors': 'Trade Creditors',
            'trade-debtors': 'Trade Debtors'
        }