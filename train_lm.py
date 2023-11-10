import hydra
import torch
from transformers import get_linear_schedule_with_warmup, get_constant_schedule
from tqdm import tqdm
import wandb

from lm.data_utils import AllDatasetsDataModule
from lm.eval_utils import evaluate
from lm.utils import (setup_wandb, get_tokenizer_from_config,
                      get_train_batch_preprocessor,
                      get_model_output_postprocessor,
                      get_model_from_config, check_tokenizer_pad_id)


@hydra.main('configs', 'wikitext2', version_base=None)
def main(args):
    print(args)

    setup_wandb(args)
    assert args.accumulated_batch_size % args.data_params.batch_size == 0
    accumulation_factor = args.accumulated_batch_size // args.data_params.batch_size

    # TODO: write parameters gathering for weight_decay
    assert args.weight_decay == 0.0

    tokenizer = get_tokenizer_from_config(args)
    check_tokenizer_pad_id(tokenizer)
    print('Tokenizer pad_token_id is: OK')
    print(f'Tokenizer vocab size: {len(tokenizer.vocab)}')
    device = torch.device('cuda:0')
    # device = torch.device('cpu')
    model = get_model_from_config(args).to(device)
    preprocessor = get_train_batch_preprocessor(args)
    postprocessor = get_model_output_postprocessor(args)

    dl_train, dl_valid, dl_test = AllDatasetsDataModule(tokenizer=tokenizer, **args.data_params).get_dataloaders()

    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    total_steps = len(dl_train) * args.epochs
    match args.scheduler:
        case None:
            sched = get_constant_schedule(opt)
        case 'triangular':
            sched = get_linear_schedule_with_warmup(opt, total_steps * args.warmup_part, total_steps)
        case _:
            raise NotImplementedError

    # eval_results = evaluate(model, dl_valid, device, 'val', preprocessor, postprocessor)
    # eval_results['epoch'] = 0
    # print('Val scores: ', eval_results)
    # wandb.log(eval_results)
    torch.autograd.set_detect_anomaly(True)

    processed_batches = 0
    losses_micro_batches = []
    for epoch in range(args.epochs):
        train_iterator = tqdm(dl_train, mininterval=10)
        for batch in train_iterator:
            batch = batch.to(device)
            input_dict = preprocessor(batch, device)
            outputs = model(**input_dict)
            outputs = postprocessor(batch, outputs)
            loss = outputs['loss']
            loss.backward()
            losses_micro_batches.append(loss.item())
            if (processed_batches + 1) % accumulation_factor == 0:
                train_iterator.set_description(f'Loss: {loss.item():.3f}')
                opt.step()
                opt.zero_grad()
                sched.step()
                mini_batch_loss = torch.mean(torch.tensor(losses_micro_batches))
                losses_micro_batches = []
                wandb.log({'loss': mini_batch_loss})
            processed_batches += 1
        eval_results = evaluate(model, dl_valid, device, 'val', preprocessor, postprocessor)
        eval_results['epoch'] = epoch + 1
        print('Val scores: ', eval_results)
        wandb.log(eval_results)
    eval_results = evaluate(model, dl_test, device, 'test', preprocessor, postprocessor)
    print('Test scores: ', eval_results)
    wandb.run.summary.update(eval_results)


if __name__ == '__main__':
    main()
