from connector import Ticker


apple_ticker = Ticker("AAPL")
stock_option_data = apple_ticker.get_stock_option_data()
print(f"Stock Option Data for Apple (AAPL): {stock_option_data}")