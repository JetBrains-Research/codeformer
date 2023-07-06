from dataclasses import dataclass
from typing import List, Optional, cast

import torch


@dataclass
class LabeledCodeTokens:
    label: List[int]
    code: torch.tensor
    num_splits: int


def transpose(list_of_lists: List[List[int]]) -> List[List[int]]:
    return [cast(List[int], it) for it in zip(*list_of_lists)]


class BatchedLabeledCodeTokens:
    def __init__(self, all_samples: List[Optional[LabeledCodeTokens]]):
        samples = [s for s in all_samples if s is not None]
        self._len = len(samples)
        self.label_tokens = torch.tensor(
            transpose([sample.label for sample in samples]), dtype=torch.long
        )
        self.code_tokens = torch.cat([sample.code for sample in samples]) if len(samples) > 0 else torch.tensor([])
        self.batch_split = torch.tensor([sample.num_splits for sample in samples])

    def __len__(self) -> int:
        return self._len

    def pin_memory(self) -> "BatchedLabeledCodeTokens":
        self.label_tokens.pin_memory()
        self.code_tokens.pin_memory()
        self.batch_split.pin_memory()
        return self

    def move_to_device(self, device: torch.device):
        self.label_tokens = self.label_tokens.to(device)
        self.code_tokens = self.code_tokens.to(device)
        self.batch_split = self.batch_split.to(device)
