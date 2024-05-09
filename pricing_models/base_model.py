from abc import ABC, abstractmethod


class PricingModel(ABC):
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def get_density_function(self) -> float:
        raise NotImplementedError

    @abstractmethod
    def get_characteristic_function(self, u, params, S0, r, q, T, model) -> float:
        raise NotImplementedError
