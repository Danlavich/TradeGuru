import unittest
from matvikk_parser import TradingEconomicsParser

class TestTradingEconomicsParser(unittest.TestCase):
    def test_parse_fundamental_data(self):
        test_html = """
        <div class="card-body" style="text-align: justify">
            <h2>Apple reported $45.91B in EBITDA for its fiscal quarter ending in December of 2024.</h2>
        </div>
        """
        parser = TradingEconomicsParser()
        result = parser.parse_fundamental_data(test_html)
        self.assertEqual(
            result,
            "Apple reported $45.91B in EBITDA for its fiscal quarter ending in December of 2024."
        )

    def test_parse_empty_data(self):
        parser = TradingEconomicsParser()
        self.assertIsNone(parser.parse_fundamental_data(""))
        self.assertIsNone(parser.parse_fundamental_data("<html></html>"))

if __name__ == "__main__":
    unittest.main()