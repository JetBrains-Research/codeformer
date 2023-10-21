from typing import List, Optional

from datasets import load_dataset
from pytorch_lightning import LightningDataModule
from tokenizers import Tokenizer
import torch
from torch.utils.data import DataLoader
from torch.utils.data import IterableDataset

from jetnn.data_processing.tree_representation.my_text_tree import MyTextTree


class ThePileDataset(IterableDataset):
    def __init__(
            self,
            split: str,
            tokenizer: Tokenizer,
            max_text_tokens: int,
            max_chunks_number: int,
            max_chunk_size: int
    ):
        super(ThePileDataset).__init__()
        self.ds = load_dataset('monology/pile-uncopyrighted', streaming=True, split=split)
        self.max_text_tokens = max_text_tokens
        self.max_chunks_number = max_chunks_number
        self.max_chunk_size = max_chunk_size
        self.tokenizer = tokenizer
        self._text_tree = MyTextTree()

    def __iter__(self):
        for sample in self.ds:
            text = sample['text']
            tokenized_text = self.tokenizer.encode(
                text,
                add_special_tokens=False,
                max_length=self.max_text_tokens,
                truncation="longest_first"
            )

            # tokenized_text = self.tokenize(sample['text'], self._config.max_text_tokens)
            # tokenized_text = list(filter(lambda x: x != self._vocab.pad_id(), tokenized_text))[1:-1]
            tokens = [self.tokenizer.decode(token) for token in tokenized_text]
            tokens_split = self._text_tree.process_text(
                text, tokens, self.max_chunk_size
            )
            num_splits = min(self.max_chunks_number, len(tokens_split))
            tokens_split = tokens_split[:num_splits]
            yield TextTokens(torch.tensor(tokenized_text), torch.tensor(tokens_split))
            tokens_splits = tokens_splits[:self.max_chunks_number]
            yield TextTokens(tokenized_text, tokens_splits)



class ThePileDataModule(LightningDataModule):
    def __init__(self,
                 batch_size: int,
                 tokenizer: Tokenizer,
                 max_text_tokens: int,
                 max_chunks_number: int,
                 max_chunk_size: int,
                 num_workers: int = 16,
                 prefetch_factor: int = 8) -> None:
        super().__init__()
        self.batch_size = batch_size
        self.tokenizer = tokenizer
        self.max_text_tokens = max_text_tokens
        self.max_chunks_number = max_chunks_number
        self.max_chunk_size = max_chunk_size
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor

    @staticmethod
    def collate_wrapper(
            batch: List[Optional[TextTokens]],
    ) -> BatchedTextTokens:
        return BatchedTextTokens(batch)

    def _create_dataset(self, split: str):
        return ThePileDataset(split,
                              self.tokenizer,
                              self.max_text_tokens,
                              self.max_chunks_number,
                              self.max_chunk_size)

    def _shared_dataloader(self, split: str) -> DataLoader:
        ds = self._create_dataset(split)
        return DataLoader(ds,
                          batch_size=self.batch_size,
                          collate_fn=self.collate_wrapper,
                          prefetch_factor=self.prefetch_factor,
                          num_workers=self.num_workers)

    def train_dataloader(self, *args, **kwargs) -> DataLoader:
        return self._shared_dataloader('train')

    def val_dataloader(self, *args, **kwargs) -> DataLoader:
        return self._shared_dataloader('validation')

    def test_dataloader(self, *args, **kwargs) -> DataLoader:
        return self._shared_dataloader('test')

    def predict_dataloader(self, *args, **kwargs) -> DataLoader:
        raise NotImplementedError

    def transfer_batch_to_device(
            self,
            batch: BatchedTextTokens,
            device: torch.device,
            dataloader_idx: int,
    ) -> BatchedTextTokens:
        batch.move_to_device(device)
        return batch
