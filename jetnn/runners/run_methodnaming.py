from argparse import ArgumentParser

import wandb
from commode_utils.callbacks import (
    ModelCheckpointWithUploadCallback,
    PrintEpochResultCallback,
)
from omegaconf import DictConfig, OmegaConf

from pytorch_lightning import seed_everything, Trainer
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    TQDMProgressBar,
)
from pytorch_lightning.loggers import WandbLogger
from jetnn.models import MethodNamingModel
from jetnn.data_processing.plain_code_method.plain_code_data_module import (
    PlainCodeDataModule,
)
from jetnn.data_processing.plain_code_ast_method.plain_code_ast_data_module import (
    PlainCodeAstDataModule,
)
import torch


def data_module_by_config(config: DictConfig):
    if config.data.type == "plain_code":
        return PlainCodeDataModule(config)
    elif config.data.type == "plain_code_ast":
        return PlainCodeAstDataModule(config)
    else:
        raise ValueError(f"Unknown data format")


def train(config: DictConfig):
    params = config.trainer

    data_module = data_module_by_config(config)
    vocab = data_module.vocabulary
    if config["checkpoint"] != "None":
        model = MethodNamingModel.load_from_checkpoint(
            checkpoint_path=config.checkpoint
        )
    else:
        model = MethodNamingModel(config, vocab)

    wandb.login(key=config.wandb.key)
    wandb_logger = WandbLogger(
        project=config.wandb.project,
        group=config.wandb.group,
        log_model=False,
        offline=config.wandb.offline,
        config=OmegaConf.to_container(config),
    )

    checkpoint_callback = ModelCheckpointWithUploadCallback(
        dirpath=wandb_logger.experiment.dir,
        filename="{epoch:02d}-val_loss={val/loss:.4f}",
        monitor="val/loss",
        every_n_epochs=params.save_every_epoch,
        save_top_k=-1,
        auto_insert_metric_name=False,
    )

    early_stopping_callback = EarlyStopping(
        patience=params.patience, monitor="val/loss", verbose=True, mode="min"
    )
    print_epoch_result_callback = PrintEpochResultCallback(after_test=False)
    lr_logger = LearningRateMonitor("step")
    progress_bar = TQDMProgressBar(refresh_rate=config.progress_bar_refresh_rate)

    if torch.cuda.is_available():
        trainer = Trainer(
            max_epochs=params.n_epochs,
            gradient_clip_val=params.clip_norm,
            deterministic=True,
            check_val_every_n_epoch=params.val_every_epoch,
            log_every_n_steps=params.log_every_n_steps,
            logger=wandb_logger,
            accelerator="gpu",
            devices=1,
            callbacks=[
                lr_logger,
                early_stopping_callback,
                checkpoint_callback,
                print_epoch_result_callback,
                progress_bar,
            ],
        )
    else:
        trainer = Trainer(
            max_epochs=params.n_epochs,
            gradient_clip_val=params.clip_norm,
            deterministic=True,
            check_val_every_n_epoch=params.val_every_epoch,
            log_every_n_steps=params.log_every_n_steps,
            logger=wandb_logger,
            callbacks=[
                lr_logger,
                early_stopping_callback,
                checkpoint_callback,
                print_epoch_result_callback,
                progress_bar,
            ],
        )

    trainer.fit(model=model, datamodule=data_module)
    trainer.test(model=model, datamodule=data_module)


def test(config: DictConfig):
    if config["checkpoint"] == "None":
        raise RuntimeError("Wrong config: No checkpoint path")

    data_module = data_module_by_config(config)
    model = MethodNamingModel.load_from_checkpoint(
        checkpoint_path=config.checkpoint, config=config, vocab=data_module.vocabulary
    )
    if torch.cuda.is_available():
        trainer = Trainer(accelerator="gpu", devices=1)
    else:
        trainer = Trainer()
    trainer.test(model, datamodule=data_module)


if __name__ == "__main__":
    torch.set_float32_matmul_precision('medium')
    arg_parser = ArgumentParser()
    arg_parser.add_argument(
        "mode", help="Mode to run script", choices=["train", "test"]
    )
    arg_parser.add_argument(
        "-c", "--config", help="Path to YAML configuration file", type=str
    )
    args = arg_parser.parse_args()

    config = OmegaConf.load(args.config)
    seed_everything(config.seed)

    if args.mode == "train":
        train(config)
    elif args.mode == "test":
        test(config)
