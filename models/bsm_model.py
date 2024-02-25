from base_model import PricingMethod
from scipy import stats
import numpy as np


class BSM(PricingMethod):
    """Black-Scholes-Merton model"""

    def __init__(self) -> None:
        super().__init__()

    def _calculate_call_option_price(
        self, S: float, K: float, T: float, r: float, sigma: float
    ) -> float:
        """Calculates option price for call option.

        Args:
            S (float): Today's price of the asset
            K (float): Strike price
            T (float): maturity (in years)
            r (float): risk free interest rate
            sigma (float): volatility

        Returns:
            float: Today's premium for a call option

        References: https://www.investopedia.com/terms/b/blackscholes.asp#:~:text=The%20Black%2DScholes%20model%2C%20aka,free%20rate%2C%20and%20the%20volatility.
        """
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = (np.log(S / K) + (r - 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        BS_C = S * stats.norm.cdf(d1, 0.0, 1.0) - K * np.exp(-r * T) * stats.norm.cdf(
            d2, 0.0, 1.0
        )
        return BS_C

    def _calculate_put_option_price(self, S, K, T, r, sigma):
        return NotImplementedError

if __name__ == "__main__":
    S = 182.52
    K = 200.77
    T = 5
    r = 0
    sigma = 0.1474
    method = BSM()
    price = method.price(option_type="call", S=S, K=K, T=T, r=r, sigma=sigma)
    print(price)