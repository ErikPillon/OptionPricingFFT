import requests
from datetime import datetime, timedelta
import json

import yfinance as YF


class YahooFinanceConnector:
    @staticmethod
    def get_stock_data(symbol, start_date, end_date, interval="1d", events="history"):
        return YF.Ticker(symbol).history(start=start_date, end=end_date)


class Ticker:
    CACHE_EXPIRATION_TIME = timedelta(minutes=180)

    def __init__(self, symbol):
        self.symbol = symbol
        self._cached_data = None
        self._last_cache_time = None

    def get_stock_option_data(self):
        if self._cached_data is None or self._is_cache_expired():
            # Connect to Yahoo Finance and query stock option data
            start_date = datetime.now() - timedelta(days=365 * 5)
            end_date = datetime.now()
            stock_data = YahooFinanceConnector.get_stock_data(
                self.symbol, start_date, end_date
            )

            # Cache the result
            self._cached_data = stock_data
            self._last_cache_time = datetime.now()

        return self._cached_data

    def _is_cache_expired(self):
        if self._last_cache_time is None:
            return True
        return datetime.now() - self._last_cache_time > Ticker.CACHE_EXPIRATION_TIME


# Example usage:
if __name__ == "__main__":
    apple_ticker = Ticker("AAPL")
    stock_option_data = apple_ticker.get_stock_option_data()
    print(f"Stock Option Data for Apple (AAPL): {stock_option_data}")
