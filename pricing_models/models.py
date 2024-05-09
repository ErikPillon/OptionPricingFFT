from pricing_models.base_model import PricingModel
import numpy as np


class VG(PricingModel):
    def __init__(
        self, sigma: float = 0.2, nu: float = 0.1, theta: float = 0.05
    ) -> None:
        super().__init__()
        self.sigma = sigma
        self.nu = nu
        self.theta = theta

    def get_density_function(self) -> float:
        return NotImplementedError

    def get_characteristic_function(self) -> float:
        return NotImplementedError


class Heston(PricingModel):
    def __init__(
        self,
        kappa: float = 0.5,
        theta: float = 0.05,
        sigma: float = 0.2,
        rho: float = 0.5,
        v0: float = 0.05,
    ) -> None:
        super().__init__()
        self.kappa = kappa
        self.theta = theta
        self.sigma = sigma
        self.rho = rho
        self.v0 = v0

    def get_density_function(self) -> float:
        return NotImplementedError

    def get_characteristic_function(self) -> float:
        return NotImplementedError


class BSM(PricingModel):
    def __init__(self, sig: float = 0.2) -> None:
        super().__init__()
        self.sig = sig

    def get_density_function(
        self, S: float, S0: float, T: float, r: float, q: float, sig: float
    ) -> float:
        """Computes the log normal density function for the stock price."""
        f = np.exp(
            -0.5
            * ((np.log(S / S0) - (r - q - sig**2 / 2) * T) / (sig * np.sqrt(T))) ** 2
        ) / (sig * S * np.sqrt(2 * np.pi * T))
        return f

    def get_characteristic_function(
        self, u, S0: float, r: float, q: float, T: float
    ) -> float:
        mu = np.log(S0) + (r - q - self.sig**2 / 2) * T
        a = self.sig * np.sqrt(T)
        phi = np.exp(1j * mu * u - (a * u) ** 2 / 2)

        return phi
