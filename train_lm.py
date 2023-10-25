import hydra
import torch
from torch import nn, Tensor
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from tqdm import tqdm
import wandb

from lm.model import CodeformerLM
from lm.data_utils import ThePileDataModule, WikiText2RawDataModule
from lm.utils import setup_wandb


@torch.no_grad()
def evaluate(model: nn.Module,
             dl: DataLoader,
             device: torch.DeviceObjType,
             split: str) -> dict[Tensor]:
    results = []
    for batch in dl:
        batch = batch.to(device)
        outputs = model(batch)
        results.append(outputs)
    log_probs_sum = 0
    total_num_tokens = 0
    total_loss = 0
    total_samples = 0
    for res in results:
        log_probs_sum = log_probs_sum + res['loss'] * res['num_tokens']
        total_num_tokens = total_num_tokens + res['num_tokens']
        total_loss = total_loss + res['loss'] * res['batch_size']
        total_samples = total_samples + res['batch_size']
    ppl = log_probs_sum / total_num_tokens
    loss = total_loss / total_samples
    logs = {'ppl': ppl, 'loss': loss}
    return {f'{split}_{key}': val for key, val in logs.items()}


@hydra.main('configs', 'wikitext2', version_base=None)
def main(args):
    print(args)

    setup_wandb(args)
    assert args.batch_size % args.micro_batch_size == 0
    accumulation_factor = args.batch_size // args.micro_batch_size

    # TODO: write parameters gathering for weight_decay
    assert args.weight_decay == 0.0
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

    dl_train = dm.train_dataloader()
    dl_valid = dm.val_dataloader()
    dl_test = dm.val_dataloader()

    device = torch.device('cuda:0')
    model = CodeformerLM(args.model_name).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    eval_results = evaluate(model, dl_valid, device, 'val')
    eval_results['epoch'] = 0
    wandb.log(eval_results)

    train_iterator = tqdm(dl_train)
    processed_batches = 0
    losses_micro_batches = []
    for epoch in range(args.epochs):
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
        eval_results = evaluate(model, dl_valid, device, 'val')
        eval_results['epoch'] = epoch + 1
        print('Val scores: ', eval_results)
        wandb.log(eval_results)
    eval_results = evaluate(model, dl_test, device, 'test')
    print('Test scores: ', eval_results)
    wandb.run.summary.update(eval_results)


if __name__ == '__main__':
    main()
