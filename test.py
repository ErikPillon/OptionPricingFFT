from abc import ABC, abstractmethod

# from models.base_model import PricingModel
import numpy as np
from scipy import stats


class PricingModel(ABC):
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def get_density_function(self) -> float:
        raise NotImplementedError

    @abstractmethod
    def get_characteristic_function(self, u, params, S0, r, q, T, model) -> float:
        raise NotImplementedError


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


class Method(ABC):
    def __init__(
        self,
        model: PricingModel,
        K: float = 100.63,
        S0: float = 180.49,
        T: float = 5,
        r: float = 0.02,
        q: float = 0.005,
        sig: float = 0.275,
    ) -> None:
        self.model = model
        self.K = K  # strike price
        self.S0 = S0  # current price
        self.T = T  # maturity period
        self.r = r  # risk free rate
        self.q = q  # dividend yield
        self.sig = sig  # volatility

    @abstractmethod
    def get_put(self) -> float:
        raise NotImplementedError

    def get_call(self) -> float:
        raise NotImplementedError

    def get_discount_factor(self) -> float:
        return np.exp(-self.r * self.T)


class BruteForce(Method):
    def __init__(self, model: PricingModel, n: int = 12) -> None:
        super().__init__(model)
        self.N = 2**n

    def get_put(self) -> float:
        # discount factor
        df = np.exp(-self.r * self.T)
        # step size
        eta = 1.0 * self.K / self.N
        # vector of stock prices
        S = np.arange(0, self.N) * eta
        # avoid numerical issues
        S[0] = 1e-8
        # vector of weights
        w = np.ones(self.N) * eta
        w[0] = eta / 2

        distributions = self.model.get_density_function(
            S=S, S0=self.S0, T=self.T, r=self.r, q=self.q, sig=self.sig
        )

        # numerical integral
        sumP = np.sum((self.K - S) * distributions * w)
        # price put
        priceP = df * sumP
        return priceP

    def get_call(self) -> float:
        # discount factor
        df = np.exp(-self.r * self.T)
        # step size
        eta = (20 * self.K - self.K) / self.N
        # vector of stock prices
        S = np.arange(self.K, 20 * self.K, eta)
        # vector of weights
        w = np.ones(self.N) * eta
        w[0] = eta / 2

        distributions = self.model.get_density_function(
            S=S, S0=self.S0, T=self.T, r=self.r, q=self.q, sig=self.sig
        )

        # numerical integral
        sumP = np.sum((S - self.K) * distributions * w)
        # price put
        priceP = df * sumP
        return priceP


class Exact(Method):
    def __init__(self, model: PricingModel) -> None:
        super().__init__(model)

    def get_call(self) -> float:
        d1 = self.get_d1()
        d2 = self.get_d2()
        BS_C = self.S0 * np.exp(-self.q * self.T) * stats.norm.cdf(
            d1, 0.0, 1.0
        ) - self.K * np.exp(-self.r * self.T) * stats.norm.cdf(d2, 0.0, 1.0)
        return BS_C

    def get_put(self) -> float:
        d1 = self.get_d1()
        d2 = self.get_d2()
        return self.K * np.exp(-self.r * self.T) * stats.norm.cdf(
            -d2, 0.0, 1.0
        ) - self.S0 * np.exp(-self.q * self.T) * stats.norm.cdf(-d1, 0.0, 1.0)

    def get_d1(self) -> float:
        return (np.log(self.S0 / self.K) + (self.r + 0.5 * self.sig**2) * self.T) / (
            self.sig * np.sqrt(self.T)
        )

    def get_d2(self) -> float:
        return (np.log(self.S0 / self.K) + (self.r - 0.5 * self.sig**2) * self.T) / (
            self.sig * np.sqrt(self.T)
        )


class Fourier(Method):
    """Performs numerical integration using the Fourier method."""

    def __init__(self, model: PricingModel) -> None:
        super().__init__(model)

    def _evaluate_fourier(self) -> float:
        val = 0
        return val

    def get_call(self) -> float:
        return NotImplementedError

    def get_put(self) -> float:
        return NotImplementedError


class FFT(Method):
    """Performs numerical integration using the Fourier method.
    Varies from normal Fourier by the fact that we exploit the fast Fourier transform instead of the normal Fourier.
    """

    def __init__(self, model: PricingModel) -> None:
        super().__init__(model)

    def _evaluate_integral_with_fft(
        self,
        n: int = 10,
        eta: float = 0.01,
        alpha: float = 1.0,
    ) -> float:
        N = 2**n
        lda = (2 * np.pi / N) / eta

        # beta = np.log(S0)-N*lda/2 # the log strike we want is in the middle of the array
        beta = np.log(self.K)

        km = np.zeros(N)
        xX = np.zeros(N)

        nuJ = np.arange(N) * eta

        psi_nuJ = self.model.get_characteristic_function(
            nuJ - (alpha + 1) * 1j, self.S0, self.r, self.q, self.T
        ) / ((alpha + 1j * nuJ) * (alpha + 1 + 1j * nuJ))

        df = self.get_discount_factor()

        km = beta + lda * np.arange(N)
        w = eta * np.ones(N)
        w[0] = eta / 2
        xX = np.exp(-1j * beta * nuJ) * df * psi_nuJ * w

        yY = np.fft.fft(xX)
        # cT_km = np.zeros(N)
        multiplier = np.exp(-alpha * km) / np.pi
        cT_km = multiplier * np.real(yY)
        return km, cT_km

    def get_call(self) -> float:
        km, cT_km = self._evaluate_integral_with_fft()
        return cT_km[0]

    def get_put(self) -> float:
        km, cT_km = self._evaluate_integral_with_fft()
        return cT_km[0]


if __name__ == "__main__":
    bsm = BSM()
    bruteforce_bsm = BruteForce(model=bsm)
    exact_bsm = Exact(model=bsm)
    fourier = Fourier(model=bsm)
    print("call prices:")
    print("bruteforce=", bruteforce_bsm.get_call())
    print("exact=", exact_bsm.get_call())
    # print("fourier=", fourier.get_call())
    print("fft=", FFT(model=bsm).get_call())
    # print(method.get_call())
    print("put prices:")
    print("bruteforce=", bruteforce_bsm.get_put())
    print("exact=", exact_bsm.get_put())
    # print("fourier=", fourier.get_put())
    print("fft=", FFT(model=bsm).get_put())
