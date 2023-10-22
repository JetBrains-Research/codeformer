import torch
from transformers import AutoTokenizer
from tqdm import tqdm

from lm.model import CodeformerLM
from lm.data_util import ThePileDataModule


def main():
    batch_size = 2
    model_name = 'microsoft/deberta-v3-base'

    max_text_tokens = 2048
    max_chunk_size = 14
    max_chunks_number = 384
    weight_decay = 0.0

    # TODO: write parameters gathering for weight_decay
    assert weight_decay == 0.0

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    dm = ThePileDataModule(batch_size, tokenizer, max_text_tokens,
                           max_chunks_number, max_chunk_size, num_workers=0,
                           prefetch_factor=None)
    dl = dm.train_dataloader()
    train_iterator = tqdm(dl)

    device = torch.device('cuda:0')
    model = CodeformerLM(model_name).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.0)

    for batch in train_iterator:
        opt.zero_grad()
        batch = batch.to(device)
        loss_tens = model(batch)
        loss = loss_tens.sum() / torch.count_nonzero(loss_tens)
        loss.backward()
        opt.step()


if __name__ == '__main__':
    main()
