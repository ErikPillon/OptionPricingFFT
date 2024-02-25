from datetime import datetime, timedelta

import yfinance as YF


class YahooFinanceConnector:
    @staticmethod
    def get_stock_data(symbol, start_date, end_date, interval="1d", events="history"):
        return YF.Ticker(symbol).history(start=start_date, end=end_date)


class Fetcher:
    def __init__(self, symbol):
        self.symbol = symbol

    def get_stock_option_data(self, start_date=None, end_date=None):
        # Connect to Yahoo Finance and query stock option data
        if start_date is None:
            start_date = datetime.now() - timedelta(days=365 * 5)
        if end_date is None:
            end_date = datetime.now()
        
        stock_data = YahooFinanceConnector.get_stock_data(
            self.symbol, start_date, end_date
        )

        # Cache the result
        return stock_data

# Example usage:
if __name__ == "__main__":
    fetcher = Fetcher("AAPL")
    stock_option_data = fetcher.get_stock_option_data()
    print(f"Stock Option Data for Apple (AAPL): {stock_option_data}")
