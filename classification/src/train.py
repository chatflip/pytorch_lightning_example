import os
import shutil
from typing import Any, cast

import hydra
import pytorch_lightning as L
import timm
import torch
import torch.nn as nn
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from pytorch_lightning.loggers import MLFlowLogger

from ClassificationDataModule import ClassificationDataModule
from ImageClassifier import ImageClassifier
from utils import ElapsedTimePrinter


def convert_script_model(args: DictConfig, model: nn.Module) -> None:
    """モデルをTorchScript形式に変換して保存する

    Args:
        args: weight_root、exp_name、model_nameを含む設定引数。
        model: 変換するニューラルネットワークモデル。
    """
    model_path = os.path.join(args.weight_root, f"{args.exp_name}_{args.model_name}.pt")
    model.eval()
    script_model = torch.jit.script(model)
    torch.jit.save(script_model, model_path)


@hydra.main(version_base=None, config_path="../config", config_name="base")
def main(args: DictConfig) -> None:
    """メイン関数

    Args:
        args: 設定引数。
    """
    print(OmegaConf.to_yaml(args))
    L.seed_everything(args.seed)

    mlf_logger = MLFlowLogger(
        experiment_name="1_classification_food101",
        log_model=True,
    )
    hyperparams = cast(dict[str, Any], OmegaConf.to_container(args, resolve=True))
    mlf_logger.log_hyperparams(hyperparams)

    model = timm.create_model(
        model_name=args.model_name, pretrained=True, num_classes=args.num_classes
    )
    model_cfg = cast(dict[str, object], model.pretrained_cfg)
    print(f"model_cfg: {model_cfg}")
    datamodule = ClassificationDataModule(args, model_cfg)
    criterion = nn.CrossEntropyLoss()
    plmodel = ImageClassifier(args, model, criterion)

    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        mode="min",
        dirpath=to_absolute_path(args.weight_root),
        filename=f"{args.exp_name}_{args.model_name}_best",
    )
    callbacks = [TQDMProgressBar(args.print_freq), checkpoint_callback]
    trainer = L.Trainer(
        logger=mlf_logger,
        accelerator="gpu",
        devices=1,
        max_epochs=args.epochs,
        log_every_n_steps=args.log_freq,
        strategy="ddp",
        precision=16 if args.apex else 32,
        deterministic=True,
        num_sanity_val_steps=0,
        callbacks=callbacks,
    )
    timer = ElapsedTimePrinter()
    timer.start()
    trainer.fit(plmodel, datamodule=datamodule)
    timer.end()
    trainer.test(plmodel, datamodule=datamodule, verbose=True)

    convert_script_model(args, model)
    local_ckpt_path = os.path.join(
        args.weight_root, f"{args.exp_name}_{args.model_name}_best_local.ckpt"
    )
    if os.path.exists(local_ckpt_path):
        os.remove(local_ckpt_path)
    shutil.copyfile(checkpoint_callback.best_model_path, local_ckpt_path)


if __name__ == "__main__":
    main()
