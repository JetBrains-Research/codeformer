import wandb


def setup_wandb(args=None):
    # mode = 'disabled' if args['dbg'] else None
    # run = wandb.init(project="codeformer", mode=mode)
    run = wandb.init(project="codeformer")
    # wandb.run.name = args['name']
    wandb.run.name = 'test'
    # wandb.config.update(args)
    return run
