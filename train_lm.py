import torch
from transformers import AutoTokenizer
from tqdm import tqdm
import wandb

from lm.model import CodeformerLM
from lm.data_utils import ThePileDataModule
from lm.utils import setup_wandb


def main():
    setup_wandb()
    batch_size = 64
    micro_batch_size = 4
    assert batch_size % micro_batch_size == 0
    accumulation_factor = batch_size // micro_batch_size
    model_name = 'microsoft/deberta-v3-small'

    max_text_tokens = 512
    max_chunk_size = 14
    max_chunks_number = 32
    weight_decay = 0.0
    learning_rate = 3e-5

    # TODO: write parameters gathering for weight_decay
    assert weight_decay == 0.0

    min_tokens = 64
    min_chunks = 8

    num_workers_dl = 16
    prefetch_factor_dl = 32
    # num_workers_dl = 0
    # prefetch_factor_dl = None

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    dm = ThePileDataModule(micro_batch_size, tokenizer, max_text_tokens,
                           max_chunks_number, max_chunk_size,
                           min_tokens, min_chunks,
                           num_workers=num_workers_dl,
                           prefetch_factor=prefetch_factor_dl)

    dl = dm.train_dataloader()
    train_iterator = tqdm(dl)

    device = torch.device('cuda:0')
    model = CodeformerLM(model_name).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    processed_batches = 0
    for batch in train_iterator:
        batch = batch.to(device)
        loss_tens = model(batch)
        loss = loss_tens.sum() / torch.count_nonzero(loss_tens)
        loss.backward()
        wandb.log({'loss': loss})
        train_iterator.set_description(f'Loss: {loss.item():.3f}')
        if (processed_batches + 1) % accumulation_factor == 0:
            opt.step()
            opt.zero_grad()
        processed_batches += 1


if __name__ == '__main__':
    main()
