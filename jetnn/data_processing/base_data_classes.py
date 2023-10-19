from dataclasses import dataclass
import torch
from typing import Optional, List, cast

@dataclass
class SampleData():
    text_tokens: torch.tensor
    label_tokens: Optional[List[int]]
    split: Optional[torch.tensor]


def transpose(list_of_lists: List[List[int]]) -> List[List[int]]:
    return [cast(List[int], it) for it in zip(*list_of_lists)]


class BatchedData():
    
    def __init__(self, all_samples: List[Optional[SampleData]]):
        samples: List[SampleData] = [s for s in all_samples if s is not None]
        self._len = len(samples)
        self.text_tokens: torch.tensor = self._accumulate_text_tokens(samples)
        self.label_tokens: torch.tensor = self._accumulate_label_tokens(samples)
        self.batch_split: torch.tensor = self._accumulate_batch_splits(samples)

    def __len__(self) -> int:
        return self._len

    def pin_memory(self) -> "BatchedData":
        self.label_tokens.pin_memory()
        self.text_tokens.pin_memory()
        self.batch_split.pin_memory()
        return self

    def move_to_device(self, device: torch.device) -> None:
        self.label_tokens = self.label_tokens.to(device)
        self.text_tokens = self.text_tokens.to(device)
        self.batch_split = self.batch_split.to(device)

    def _accumulate_text_tokens(self, samples: List[SampleData]) -> None:
        return torch.stack([sample.text_tokens for sample in samples]) if self._len > 0 else torch.tensor([])

    def _accumulate_label_tokens(self, samples: List[SampleData]) -> None:
        labels = [sample.label_tokens for sample in samples if sample.label_tokens is not None]
        return torch.tensor(
            transpose([label for label in labels]), dtype=torch.long
        )

    def _accumulate_batch_splits(self, samples: List[SampleData]) -> None:
        splits = [sample.split for sample in samples if sample.split is not None]
        return torch.stack([batch_split for batch_split in splits]) if len(splits) > 0 else torch.tensor([])
