import hydra
import torch
from transformers import AutoTokenizer
from tqdm import tqdm
import wandb

from lm.model import CodeformerLM
from lm.data_utils import ThePileDataModule, WikiText2RawDataModule
from lm.utils import setup_wandb

@hydra.main('configs', 'wikitext2', version_base=None)
def main(args):
    print(args)

    setup_wandb(args)
    args.batch_size = 128
    args.micro_batch_size = 4
    assert args.batch_size % args.micro_batch_size == 0
    accumulation_factor = args.batch_size // args.micro_batch_size

    args.max_text_tokens = 512
    args.max_chunk_size = 14
    args.max_chunks_number = 32
    args.weight_decay = 0.0
    args.learning_rate = 1e-4

    # TODO: write parameters gathering for weight_decay
    assert args.weight_decay == 0.0

    args.min_tokens = 64
    args.min_chunks = 8

    args.num_workers_dl = 16
    args.prefetch_factor_dl = 32
    # num_workers_dl = 0
    # prefetch_factor_dl = None

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    # dm = ThePileDataModule(args.micro_batch_size, tokenizer, args.max_text_tokens,
    #                        args.max_chunks_number, args.max_chunk_size,
    #                        args.min_tokens, args.min_chunks,
    #                        num_workers=args.num_workers_dl,
    #                        prefetch_factor=args.prefetch_factor_dl)
    dm = WikiText2RawDataModule(args.micro_batch_size, tokenizer, args.max_text_tokens,
                                args.max_chunks_number, args.max_chunk_size,
                                args.min_tokens, args.min_chunks,
                                num_workers=args.num_workers_dl,
                                prefetch_factor=args.prefetch_factor_dl)

    dl = dm.train_dataloader()
    train_iterator = tqdm(dl)

    device = torch.device('cuda:0')
    model = CodeformerLM(args.model_name).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    processed_batches = 0
    losses_micro_batches = []
    for batch in train_iterator:
        batch = batch.to(device)
        outputs = model(batch)
        loss = outputs['loss']
        loss.backward()
        losses_micro_batches.append(loss.item())
        if (processed_batches + 1) % accumulation_factor == 0:
            train_iterator.set_description(f'Loss: {loss.item():.3f}')
            opt.step()
            opt.zero_grad()
            mini_batch_loss = torch.mean(torch.tensor(losses_micro_batches))
            losses_micro_batches = []
            wandb.log({'loss': mini_batch_loss})
        processed_batches += 1


if __name__ == '__main__':
    main()
