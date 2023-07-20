from __future__ import annotations

from abc import ABC
from abc import abstractmethod

from chromadb.config import Settings


class Server(ABC):
    @abstractmethod
    def __init__(self, settings: Settings):
        pass
