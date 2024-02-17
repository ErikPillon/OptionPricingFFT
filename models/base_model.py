from abc import ABC, abstractmethod, abstractclassmethod
from enum import Enum


class OPTION_TYPE(Enum):
    CALL_OPTION = "call"
    PUT_OPTION = "put"


# Abstract class for pricing methods
class PricingMethod(ABC):
    """Abstract class defining interface for option pricing models."""

    def __init__(self) -> None:
        return

    def price(self, option_type, *args, **kwargs):
        """Calculates call/put option price according to the specified parameter."""
        if option_type == OPTION_TYPE.CALL_OPTION.value:
            return self._calculate_call_option_price(*args, **kwargs)
        elif option_type == OPTION_TYPE.PUT_OPTION.value:
            return self._calculate_put_option_price(*args, **kwargs)
        else:
            raise Exception("Wrong option type")

    @abstractclassmethod
    def _calculate_call_option_price(self, *args, **kwargs):
        """Calculates option price for call option."""
        raise NotImplementedError

    @abstractclassmethod
    def _calculate_put_option_price(self, *args, **kwargs):
        return NotImplementedError
