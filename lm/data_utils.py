from abc import ABC, abstractmethod
from typing import List, Optional
from dataclasses import dataclass

from datasets import load_dataset
from pytorch_lightning import LightningDataModule
from tokenizers import Tokenizer
import torch
from torch import Tensor, LongTensor
from torch.utils.data import DataLoader, IterableDataset, Dataset
import torch.multiprocessing

from jetnn.data_processing.tree_representation.my_text_tree import MyTextTree

# This fixes "RuntimeError: Too many open files. Communication with the workers is no longer possible"
torch.multiprocessing.set_sharing_strategy('file_system')


@dataclass
class TextTokens:
    token_ids: list[int] # can be code, not just text
    chunk_sizes: list[int]

    @property
    def num_splits(self) -> int:
        return len(self.chunk_sizes)

    @property
    def max_tokens_per_chunk(self) -> int:
        return max(self.chunk_sizes)


class BatchedTextTokens:
    token_ids: LongTensor
    token_ids_list: list[list[int]]
    token_ids_bos_eos: LongTensor
    token_ids_chunk: LongTensor
    token_ids_chunk_bos_eos: LongTensor
    token_ids_chunk_stacked: LongTensor
    token_ids_chunk_stacked_bos_eos: LongTensor
    chunk_sizes_list: list[list[int]]
    chunk_sizes_tensor: LongTensor
    max_tokens_per_chunk: int
    max_tokens_per_sample: int
    max_chunks_per_sample: int
    max_tokens_per_chunk_bos_eos: int
    max_tokens_per_sample_bos_eos: int
    num_chunks_total: int
    token_ids_list: list[list[int]]
    pad_token_id: int
    bos_token_id: int
    eos_token_id: int
    att_mask: Tensor
    att_mask_chunk_tokens: Tensor
    att_mask_chunks: Tensor
    batch_size: int
    chunk_lens_stacked: LongTensor
    decoder_inp_tok_ids: LongTensor
    decoder_targ_tok_ids: LongTensor

    def __init__(self,
                 samples: List[TextTokens],
                 pad_token_id: int,
                 bos_token_id: int,
                 eos_token_id: int,
                 num_previous_chunks: int) -> None:
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.num_previous_chunks = num_previous_chunks
        self.max_tokens_per_chunk = max(s.max_tokens_per_chunk for s in samples)
        self.max_tokens_per_chunk_bos_eos = self.max_tokens_per_chunk + 2
        self.max_tokens_per_sample = max(len(s.token_ids) for s in samples)
        self.max_tokens_per_sample_bos_eos = self.max_tokens_per_sample + 2
        self.max_chunks_per_sample = max(s.num_splits for s in samples)
        self.num_chunks_total = sum(s.num_splits for s in samples)
        batch_size = len(samples)
        self.batch_size = batch_size
        self.token_ids_list = [s.token_ids for s in samples]
        # self.chunk_sizes_list = [s.chunk_sizes for s in samples]
        self.token_ids = torch.full((batch_size, self.max_tokens_per_sample),
                                    pad_token_id,
                                    dtype=torch.long)
        self.token_ids_bos_eos = torch.full((batch_size, self.max_tokens_per_sample_bos_eos),
                                            pad_token_id,
                                            dtype=torch.long)
        self.token_ids_chunk = torch.full([batch_size, self.max_chunks_per_sample, self.max_tokens_per_chunk],
                                          pad_token_id, dtype=torch.long)
        self.token_ids_chunk_bos_eos = torch.full([batch_size, self.max_chunks_per_sample, self.max_tokens_per_chunk_bos_eos],
                                                  pad_token_id, dtype=torch.long)
        self.token_ids_chunk_stacked = torch.full([self.num_chunks_total, self.max_tokens_per_chunk],
                                                  pad_token_id, dtype=torch.long)
        self.token_ids_chunk_stacked_bos_eos = torch.full([self.num_chunks_total, self.max_tokens_per_chunk + 2],
                                                          pad_token_id, dtype=torch.long)
        count = 0
        self.chunk_lens_stacked = torch.zeros(self.num_chunks_total, dtype=torch.long)
        for sample_num, sample in enumerate(samples):
            n_tokens_per_sample = len(sample.token_ids)
            n_tokens_per_sample_bos_eos = n_tokens_per_sample + 2
            sample_token_ids = sample.token_ids
            self.token_ids[sample_num, :n_tokens_per_sample] = torch.tensor(sample_token_ids, dtype=torch.long)
            sample_token_ids_bos_eos = [self.bos_token_id] + sample.token_ids + [self.eos_token_id]
            self.token_ids_bos_eos[sample_num, :n_tokens_per_sample_bos_eos] = torch.tensor(sample_token_ids_bos_eos, dtype=torch.long)
            cursor = 0
            for split_num, split_size in enumerate(sample.chunk_sizes):
                chunk_tokens = sample.token_ids[cursor: cursor + split_size]
                self.token_ids_chunk_stacked[count, :len(chunk_tokens)] = torch.tensor(chunk_tokens, dtype=torch.long)
                self.chunk_lens_stacked[count] = len(chunk_tokens)
                self.token_ids_chunk[sample_num, split_num, :split_size] = torch.tensor(chunk_tokens, dtype=torch.long)
                chunk_tokens_bos_eos = [bos_token_id] + sample.token_ids[cursor: cursor + split_size] + [eos_token_id]
                self.token_ids_chunk_bos_eos[sample_num, split_num, :split_size + 2] = torch.tensor(chunk_tokens_bos_eos, dtype=torch.long)
                self.token_ids_chunk_stacked_bos_eos[count, :len(chunk_tokens_bos_eos)] = torch.tensor(chunk_tokens_bos_eos, dtype=torch.long)
                cursor += split_size
                count += 1
        self.att_mask = (self.token_ids != torch.scalar_tensor(self.pad_token_id)).float()
        self.att_mask_bos_eos = (self.token_ids_bos_eos != torch.scalar_tensor(self.pad_token_id)).float()
        self.att_mask_chunk_tokens = (self.token_ids_chunk != torch.scalar_tensor(self.pad_token_id)).float()
        self.att_mask_chunk_tokens_stacked = (self.token_ids_chunk_stacked != torch.scalar_tensor(self.pad_token_id)).float()
        self.att_mask_chunk_tokens_bos_eos = (self.token_ids_chunk_bos_eos != torch.scalar_tensor(self.pad_token_id)).float()
        self.att_mask_chunk_tokens_stacked_bos_eos = (self.token_ids_chunk_stacked_bos_eos != torch.scalar_tensor(self.pad_token_id)).float()
        self.att_mask_chunks = torch.any(self.token_ids_chunk != torch.scalar_tensor(self.pad_token_id), 2)
        self.att_mask_chunks_bos_eos = torch.any(self.token_ids_chunk_bos_eos != torch.scalar_tensor(self.pad_token_id), 2)

        self.chunk_sizes_list = []
        self.chunk_sizes_list_bos_eos = []
        for s in samples:
            self.chunk_sizes_list.append([split_size for split_size in s.chunk_sizes])
            self.chunk_sizes_list_bos_eos.append([split_size + 2 for split_size in s.chunk_sizes])
        self.chunk_sizes_tensor = torch.zeros(batch_size, self.max_chunks_per_sample, dtype=torch.long)
        self.chunk_sizes_tensor_bos_eos = torch.zeros(batch_size, self.max_chunks_per_sample, dtype=torch.long)
        for n in range(batch_size):
            cur_chunk_sizes = torch.tensor(self.chunk_sizes_list[n], dtype=torch.long)
            self.chunk_sizes_tensor[n, : len(cur_chunk_sizes)] = cur_chunk_sizes
            self.chunk_sizes_tensor_bos_eos[n, : len(cur_chunk_sizes)] = cur_chunk_sizes + 2

        self.decoder_inp_tok_ids, self.decoder_targ_tok_ids = self._get_decoder_inputs(self.token_ids_list,
                                                                                       self.chunk_sizes_list,
                                                                                       self.pad_token_id,
                                                                                       self.bos_token_id,
                                                                                       self.eos_token_id,
                                                                                       self.num_previous_chunks)
        self.decoder_tok_mask = (self.decoder_inp_tok_ids != self.pad_token_id).float()

    def __len__(self) -> int:
        return len(self.token_ids_list)

    def to(self, device: torch.DeviceObjType) -> "BatchedTextTokens":
        self.token_ids_bos_eos = self.token_ids_bos_eos.to(device)
        self.att_mask_bos_eos = self.att_mask_bos_eos.to(device)
        self.token_ids_chunk_bos_eos = self.token_ids_chunk_bos_eos.to(device)
        self.att_mask_chunk_tokens = self.att_mask_chunk_tokens.to(device)
        self.token_ids_chunk_stacked_bos_eos = self.token_ids_chunk_stacked_bos_eos.to(device)
        self.chunk_sizes_tensor = self.chunk_sizes_tensor.to(device)
        self.att_mask_chunks = self.att_mask_chunks.to(device)
        self.decoder_inp_tok_ids = self.decoder_inp_tok_ids.to(device)
        self.decoder_targ_tok_ids = self.decoder_targ_tok_ids.to(device)
        return self

    def _get_decoder_inputs(self,
                            token_ids_list: list[list[int]],
                            chunk_sizes_list: list[list[int]],
                            pad_token_id: int,
                            bos_token_id: int,
                            eos_token_id: int,
                            num_previous_chunks: int) -> tuple[LongTensor, LongTensor]:
        decoder_inp_list = []
        decoder_targ_list = []
        for sample_num, chunk_sizes in enumerate(chunk_sizes_list):
            num_chunks = len(chunk_sizes)
            for chunk_num in range(num_chunks):
                chunk_nums = list(range(chunk_num - num_previous_chunks, chunk_num + 1))
                ids, num_toks_in_prev_chunks = self._get_multiple_chunk_token_ids(token_ids_list,
                                                                                  chunk_sizes_list,
                                                                                  sample_num,
                                                                                  chunk_nums)
                targets = [pad_token_id] * num_toks_in_prev_chunks + ids[num_toks_in_prev_chunks:] + [eos_token_id]
                ids = [bos_token_id] + ids
                assert len(targets) == len(ids)
                decoder_inp_list.append(ids)
                decoder_targ_list.append(targets)
        num_chunks_total = len(decoder_inp_list)
        max_len = max(len(ids) for ids in decoder_inp_list)
        decoder_inp_token_ids = torch.full((num_chunks_total, max_len), pad_token_id, dtype=torch.long)
        decoder_targ_token_ids = decoder_inp_token_ids.detach().clone()
        for n_chunk, (inp_ids, targ_ids) in enumerate(zip(decoder_inp_list, decoder_targ_list)):
            n_toks = len(inp_ids)
            decoder_inp_token_ids[n_chunk, :n_toks] = torch.tensor(inp_ids, dtype=torch.long)
            decoder_targ_token_ids[n_chunk, :n_toks] = torch.tensor(targ_ids, dtype=torch.long)
        return decoder_inp_token_ids, decoder_targ_token_ids

    def _get_multiple_chunk_token_ids(self,
                                      token_ids_list: list[list[int]],
                                      chunk_sizes_list: list[list[int]],
                                      sample_num: int,
                                      chunk_nums: int | list[int]) -> tuple[list[int], int]:
        if isinstance(chunk_nums, int):
            chunk_nums = [chunk_nums]
        chunk_nums = set(chunk_nums)
        sizes = chunk_sizes_list[sample_num]
        start_end_points = torch.cumsum(torch.tensor([0] + sizes, dtype=torch.long), 0)
        start_points = [sp for n, sp in enumerate(start_end_points[:-1]) if n in chunk_nums]
        end_points = [ep for n, ep in enumerate(start_end_points[1:]) if n in chunk_nums]
        start_point = start_points[0] if start_points else 0
        end_point = end_points[-1] if end_points else 0
        chunks_token_ids = token_ids_list[sample_num][start_point: end_point]
        num_toks_in_prev_chunks = start_points[-1] - start_point
        return chunks_token_ids, num_toks_in_prev_chunks


class LMDatasetBase:
    def __init__(
            self,
            split: str,
            tokenizer: Tokenizer,
            max_text_tokens: int,
            max_chunks_number: int,
            max_chunk_size: int,
            num_previous_chunks: int,
            min_chunks: int = 1,
            min_tokens: int = 1
    ):
        super().__init__()
        self.split = split
        self.max_text_tokens = max_text_tokens
        self.max_chunks_number = max_chunks_number
        self.max_chunk_size = max_chunk_size
        self.num_previous_chunks = num_previous_chunks
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
            num_previous_chunks: int,
            min_chunks: int,
            min_tokens: int
    ):
        super().__init__(split,
                         tokenizer,
                         max_text_tokens,
                         max_chunks_number,
                         max_chunk_size,
                         num_previous_chunks,
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
            num_previous_chunks: int,
            min_chunks: int,
            min_tokens: int
    ):
        super().__init__(split,
                         tokenizer,
                         max_text_tokens,
                         max_chunks_number,
                         max_chunk_size,
                         num_previous_chunks,
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
                 num_previous_chunks: int,
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
        self.num_previous_chunks = num_previous_chunks
        self.min_tokens = min_tokens
        self.min_chunks = min_chunks
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor

    def collate_wrapper(self, batch: List[Optional[TextTokens]]) -> BatchedTextTokens:
        return BatchedTextTokens(batch,
                                 self.tokenizer.pad_token_id,
                                 self.tokenizer.bos_token_id,
                                 self.tokenizer.eos_token_id,
                                 self.num_previous_chunks)

    def _create_dataset(self, split: str):
        return self.name_to_dataset[self.dataset_name](split,
                                                       self.tokenizer,
                                                       self.max_text_tokens,
                                                       self.max_chunks_number,
                                                       self.max_chunk_size,
                                                       self.num_previous_chunks,
                                                       self.min_chunks,
                                                       self.min_tokens)

    def _shared_dataloader(self, split: str) -> DataLoader:
        ds = self._create_dataset(split)
        return DataLoader(ds,
                          batch_size=self.batch_size,
                          collate_fn=self.collate_wrapper,
                          prefetch_factor=self.prefetch_factor,
                          num_workers=self.num_workers,
                          pin_memory=True)

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
