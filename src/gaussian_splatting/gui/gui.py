__all__ = ["GUI"]
from abc import ABC, abstractmethod

class GUI(ABC):
    """Base class for gui aplications.
    """
    @abstractmethod
    def start(self) -> None:
        """Starts the application.
        """