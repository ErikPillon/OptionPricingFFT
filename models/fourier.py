from .base_model import PricingMethod
from scipy import stats
import numpy as np


class StandardFourier(PricingMethod):

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
        """
        return NotImplementedError

    def _calculate_put_option_price(self, S, K, T, r, sigma):
        return NotImplementedError
    
class FastFourier(PricingMethod):

    def __init__(self) -> None:
        super().__init__()

    def _calculate_call_option_price(
        self, S: float, K: float, T: float, r: float, sigma: float
    ) -> float:
        return NotImplementedError

    def _calculate_put_option_price(self, S, K, T, r, sigma):
        return NotImplementedError

