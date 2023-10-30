from abc import ABC, abstractmethod
from typing import List, Optional
from dataclasses import dataclass

from datasets import load_dataset
from pytorch_lightning import LightningDataModule
from tokenizers import Tokenizer
import torch
from torch import Tensor
from torch.utils.data import DataLoader, IterableDataset, Dataset
import torch.multiprocessing

from jetnn.data_processing.tree_representation.my_text_tree import MyTextTree

# This fixes "RuntimeError: Too many open files. Communication with the workers is no longer possible"
torch.multiprocessing.set_sharing_strategy('file_system')


@dataclass
class TextTokens:
    token_ids: list[int] # can be code, not just text
    split_sizes: list[int]

    @property
    def num_splits(self) -> int:
        return len(self.split_sizes)

    @property
    def max_tokens_per_split(self) -> int:
        return max(self.split_sizes)


class BatchedTextTokens:
    token_ids: Tensor
    token_ids_chunk: Tensor
    split_sizes_list: list[list[int]]
    max_tokens_per_split: int
    max_tokens_per_sample: int
    max_splits: int
    token_ids_list: list[list[int]]
    pad_token_id: int
    bos_token_id: int
    eos_token_id: int
    att_mask: Tensor
    att_mask_chunk_tokens: Tensor
    att_mask_chunks: Tensor

    def __init__(self,
                 samples: List[TextTokens],
                 pad_token_id: int,
                 bos_token_id: int,
                 eos_token_id: int) -> None:
        # + 2 because of bos and eos tokens
        self.max_tokens_per_split = max(s.max_tokens_per_split for s in samples) + 2
        self.max_tokens_per_sample = max(len(s.token_ids) for s in samples) + 2
        self.max_splits = max(s.num_splits for s in samples)
        batch_size = len(samples)
        self.token_ids_list = [s.token_ids for s in samples]
        # self.split_sizes_list = [s.split_sizes for s in samples]
        self.token_ids = pad_token_id * torch.ones(batch_size,
                                                   self.max_tokens_per_sample,
                                                   dtype=torch.long)
        self.token_ids_chunk = pad_token_id * torch.ones(batch_size,
                                                         self.max_splits,
                                                         self.max_tokens_per_split,
                                                         dtype=torch.long)
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        for sample_num, sample in enumerate(samples):
            n_tokens_per_sample = len(sample.token_ids) + 2
            sample_token_ids = [self.bos_token_id] + sample.token_ids + [self.eos_token_id]
            self.token_ids[sample_num, :n_tokens_per_sample] = torch.tensor(sample_token_ids, dtype=torch.long)
            cursor = 0
            for split_num, split_size in enumerate(sample.split_sizes):
                chunk_tokens = [bos_token_id] + sample.token_ids[cursor: cursor + split_size] + [eos_token_id]
                # + 2 because of bos and eos tokens
                self.token_ids_chunk[sample_num, split_num, :split_size + 2] = torch.tensor(chunk_tokens, dtype=torch.long)
                cursor += split_size
        self.att_mask = (self.token_ids != torch.scalar_tensor(self.pad_token_id)).float()
        self.att_mask_chunk_tokens = (self.token_ids_chunk != torch.scalar_tensor(self.pad_token_id)).float()
        self.att_mask_chunks = torch.any(self.token_ids_chunk != self.pad_token_id, 2)

        self.split_sizes_list = []
        for s in samples:
            self.split_sizes_list.append([split_size + 2 for split_size in s.split_sizes])

    def __len__(self) -> int:
        return len(self.token_ids_list)

    def pin_memory(self) -> "BatchedTextTokens":
        self.token_ids.pin_memory()
        return self

    def to(self, device: torch.DeviceObjType) -> "BatchedTextTokens":
        self.token_ids = self.token_ids.to(device)
        self.token_ids_chunk = self.token_ids_chunk.to(device)
        self.att_mask = self.att_mask.to(device)
        self.att_mask_chunk_tokens = self.att_mask_chunk_tokens.to(device)
        self.att_mask_chunks = self.att_mask_chunks.to(device)
        return self


class LMDatasetBase:
    def __init__(
            self,
            split: str,
            tokenizer: Tokenizer,
            max_text_tokens: int,
            max_chunks_number: int,
            max_chunk_size: int,
            min_chunks: int = 1,
            min_tokens: int = 1
    ):
        super(ThePileDataset).__init__()
        self.split = split
        self.max_text_tokens = max_text_tokens
        self.max_chunks_number = max_chunks_number
        self.max_chunk_size = max_chunk_size
        self.tokenizer = tokenizer
        self._text_tree = MyTextTree()
        self.min_chunks = min_chunks
        self.min_tokens = min_tokens

    def parse_text(self, text: str) -> TextTokens:
        tokenized_text = self.tokenizer.encode(
            text,
            add_special_tokens=False,
            max_length=self.max_text_tokens,
            truncation="longest_first"
        )
        tokens = [self.tokenizer.convert_ids_to_tokens([token])[0] for token in tokenized_text]
        if len(tokens) < self.min_tokens:
            return None
        truncated_text = self.tokenizer.decode(tokenized_text)
        tokens_splits = self._text_tree.process_text(
            truncated_text, tokens, self.max_chunk_size
        )
        tokens_splits = tokens_splits[:self.max_chunks_number]
        if len(tokens_splits) < self.min_chunks:
            return None
        return TextTokens(tokenized_text, tokens_splits)


class ThePileDataset(LMDatasetBase, IterableDataset):
    def __init__(
            self,
            split: str,
            tokenizer: Tokenizer,
            max_text_tokens: int,
            max_chunks_number: int,
            max_chunk_size: int,
            min_chunks: int,
            min_tokens: int
    ):
        super().__init__(split,
                         tokenizer,
                         max_text_tokens,
                         max_chunks_number,
                         max_chunk_size,
                         min_chunks,
                         min_tokens)

        self.ds = load_dataset('monology/pile-uncopyrighted', streaming=True, split=split)

    def __iter__(self) -> TextTokens:
        for sample in self.ds:
            text = sample['text']
            sample = self.parse_text(text)
            if sample is not None:
                yield sample


class WikiTextDatasetBase(LMDatasetBase, Dataset, ABC):

    @property
    @abstractmethod
    def _dataset_names(self):
        ...

    # Fix it to be consistent with evaluation from other papers
    def __init__(
            self,
            split: str,
            tokenizer: Tokenizer,
            max_text_tokens: int,
            max_chunks_number: int,
            max_chunk_size: int,
            min_chunks: int,
            min_tokens: int
    ):
        super().__init__(split,
                         tokenizer,
                         max_text_tokens,
                         max_chunks_number,
                         max_chunk_size,
                         min_chunks,
                         min_tokens)

        ds_raw = load_dataset(*self._dataset_names)[split]
        self.ds = []
        current_sample_texts = []
        for sample in ds_raw:
            text = sample['text']
            striped_text = text.lstrip()
            if not striped_text or striped_text[0] == '=':
                if current_sample_texts:
                    sample_text = '\n'.join(current_sample_texts)
                    self.ds.append(sample_text)
                    current_sample_texts = []
            else:
                current_sample_texts.append(text)
        if current_sample_texts:
            self.ds.append('\n'.join(current_sample_texts))

    def __getitem__(self, idx: int) -> TextTokens:
        text = self.ds[idx]
        sample = self.parse_text(text)
        return sample

    def __len__(self) -> int:
        return len(self.ds)


class WikiText2Dataset(WikiTextDatasetBase):
    _dataset_names = ['wikitext', 'wikitext-2-v1']


class WikiText103Dataset(WikiTextDatasetBase):
    _dataset_names = ['wikitext', 'wikitext-103-v1']


class WikiText2RawDataset(WikiTextDatasetBase):
    # NOTE: This dataset is not tokenized!
    # To compare perplexity with other researchers you need to use
    # WikiText2Dataset not this one
    _dataset_names = ['wikitext', 'wikitext-2-raw-v1']


class WikiText103RawDataset(WikiTextDatasetBase):
    # NOTE: This dataset is not tokenized!
    # To compare perplexity with other researchers you need to use
    # WikiText2Dataset not this one
    _dataset_names = ['wikitext', 'wikitext-103-raw-v1']


class AllDatasetsDataModule(LightningDataModule):
    name_to_dataset = {
        'wikitext2': WikiText2Dataset,
        'wikitext2raw': WikiText2RawDataset,
        'wikitext103': WikiText103Dataset,
        'wikitext103raw': WikiText103RawDataset,
        'the_pile': ThePileDataset
    }

    def __init__(self,
                 dataset_name: str,
                 batch_size: int,
                 tokenizer: Tokenizer,
                 max_text_tokens: int,
                 max_chunks_number: int,
                 max_chunk_size: int,
                 min_tokens: int = 1,
                 min_chunks: int = 1,
                 num_workers: int = 16,
                 prefetch_factor: int | None = 8) -> None:
        super().__init__()
        self.batch_size = batch_size
        assert dataset_name in self.name_to_dataset
        self.dataset_name = dataset_name
        self.tokenizer = tokenizer
        self.max_text_tokens = max_text_tokens
        self.max_chunks_number = max_chunks_number
        self.max_chunk_size = max_chunk_size
        self.min_tokens = min_tokens
        self.min_chunks = min_chunks
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor

    def collate_wrapper(self, batch: List[Optional[TextTokens]]) -> BatchedTextTokens:
        return BatchedTextTokens(batch,
                                 self.tokenizer.pad_token_id,
                                 self.tokenizer.bos_token_id,
                                 self.tokenizer.eos_token_id)

    def _create_dataset(self, split: str):
        return self.name_to_dataset[self.dataset_name](split,
                                                       self.tokenizer,
                                                       self.max_text_tokens,
                                                       self.max_chunks_number,
                                                       self.max_chunk_size,
                                                       self.min_chunks,
                                                       self.min_tokens)

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

    def get_dataloaders(self):
        return self.train_dataloader(), self.val_dataloader(), self.test_dataloader()
