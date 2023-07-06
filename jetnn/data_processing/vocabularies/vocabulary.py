from abc import ABC, abstractmethod
from typing import Iterable, Union, List


class Vocabulary(ABC):
    @abstractmethod
    def __len__(self) -> int:
        ...

    @abstractmethod
    def encode(self, token: str) -> List[int]:
        ...

    @abstractmethod
    def decode(self, encoded) -> str:
        ...

    @abstractmethod
    def pad_id(self) -> int:
        ...

    @abstractmethod
    def bos_id(self) -> int:
        ...

    @abstractmethod
    def eos_id(self) -> int:
        ...

    @abstractmethod
    def unk_id(self) -> int:
        ...
