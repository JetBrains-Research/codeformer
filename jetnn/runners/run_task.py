from argparse import ArgumentParser

import wandb
from commode_utils.callbacks import (
    ModelCheckpointWithUploadCallback,
    PrintEpochResultCallback,
)
from omegaconf import DictConfig, OmegaConf
from jetnn.data_processing.vocabularies.vocabulary import (
    Vocabulary,
)
from pytorch_lightning import seed_everything, Trainer
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    TQDMProgressBar,
)
from pytorch_lightning.loggers import WandbLogger
from jetnn.models.tasks.methodnaming import MethodNamingModel
from jetnn.models.tasks.language_modeling import LanguageModelingModel
from jetnn.data_processing.data_module import DataModule
import torch


def data_module_by_config(config: DictConfig, task: str) -> DataModule:
    return DataModule(task, config)


def model_by_task(task: str, config: DictConfig, vocab: Vocabulary):
    if task == "language_modeling":
        model_class = LanguageModelingModel
    elif task == "code_modeling":
        model_class = LanguageModelingModel
    elif task == "method_naming":
        model_class = MethodNamingModel
    else:
        raise RuntimeError("Wrong task name")

    if config["checkpoint"] != "None":
        model = model_class.load_from_checkpoint(
            checkpoint_path=config.checkpoint
        )
    else:
        model = model_class(config, vocab)

    return model

def train(config: DictConfig, task: str, cuda_devices: list) -> None:
    params = config.trainer

    data_module = data_module_by_config(config, task)

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
    
    accumulated_grad_batches = int(params.effective_batch_size) // int(config.train.dataloader.batch_size)
    # TODO: allow run without gpu without copy paste
    if torch.cuda.is_available():
        trainer = Trainer(
            max_epochs=params.n_epochs,
            gradient_clip_val=params.clip_norm,
            deterministic=True,
            check_val_every_n_epoch=params.val_every_epoch,
            log_every_n_steps=params.log_every_n_steps,
            logger=wandb_logger,
            accelerator="gpu",
            devices=cuda_devices,
            callbacks=[
                lr_logger,
                early_stopping_callback,
                checkpoint_callback,
                print_epoch_result_callback,
                progress_bar,
            ],
            accumulate_grad_batches=accumulated_grad_batches,
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
        accumulate_grad_batches=accumulated_grad_batches,
    )

    model = model_by_task(task, config, data_module.vocabulary)
    trainer.fit(model=model, datamodule=data_module)
    trainer.test(model=model, datamodule=data_module)


def test(config: DictConfig, task: str, cuda_devices: list) -> None:
    if config["checkpoint"] == "None":
        raise RuntimeError("Wrong config: No checkpoint path")

    data_module = data_module_by_config(config, task)
    model = model_by_task(task, config, data_module.vocabulary)
    if torch.cuda.is_available():
        trainer = Trainer(accelerator="gpu", devices=cuda_devices)
    else:
        trainer = Trainer()
    trainer.test(model, datamodule=data_module)


if __name__ == "__main__":
    torch.set_printoptions(threshold=10000)
    torch.set_float32_matmul_precision("medium")
    arg_parser = ArgumentParser()
    arg_parser.add_argument(
        "-task", help="Task name", choices=["language_modeling", "code_modeling", "method_naming"]
    )
    arg_parser.add_argument(
        "-mode", help="Mode to run script", choices=["train", "test"]
    )
    arg_parser.add_argument(
        "-c", "--config", help="Path to YAML configuration file", type=str
    )
    arg_parser.add_argument(
        "-cd", "--cuda_devices", help="available cuda devices", action='append'
    )
    args = arg_parser.parse_args()
    cuda_devices = [int(cd) for cd in args.cuda_devices]
    config = OmegaConf.load(args.config)

    seed_everything(config.seed)

    if args.mode == "train":
        train(config, args.task, cuda_devices)
    elif args.mode == "test":
        test(config, args.task, cuda_devices)
