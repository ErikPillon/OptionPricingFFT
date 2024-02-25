from fetcher import Fetcher


# apple_ticker = Ticker("AAPL")
# stock_option_data = apple_ticker.get_stock_option_data()
# print(f"Stock Option Data for Apple (AAPL): {stock_option_data}")

if __name__ == "__main__":
    fetcher = Fetcher("AAPL")
    data = fetcher.get_stock_option_data()["Close"]
    fig = data.plot()
    fig.show()