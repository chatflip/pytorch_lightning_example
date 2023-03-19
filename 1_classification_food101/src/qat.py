import os

import hydra
import pytorch_lightning as pl
import timm
import torch
import torch.nn as nn
import yaml
from ClassificationDataModule import ClassificationDataModule
from hydra.utils import to_absolute_path
from ImageClassifier import ImageClassifier
from model import mobilenet_v2
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import ModelCheckpoint, QuantizationAwareTraining
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from pytorch_lightning.loggers import MLFlowLogger
from utils import ElapsedTimePrinter


def convert_script_model(args, model):
    model = model.eval().to("cpu")
    cwd = hydra.utils.get_original_cwd()
    model_path = os.path.join(cwd, args.weight_root, f"{args.exp_name}_mobilenetv2.pt")
    script_model = torch.jit.script(model)
    torch.jit.save(script_model, model_path)


@hydra.main(version_base=None, config_path="../config", config_name="qat")
def main(args: DictConfig) -> None:
    print(OmegaConf.to_yaml(args))
    pl.seed_everything(args.seed)

    mlf_logger = MLFlowLogger(
        experiment_name="1_classification_animeface",
        log_model=True,
    )
    mlf_logger.log_hyperparams(args)

    model = timm.create_model(
        model_name=args.model_name, pretrained=True, num_classes=args.num_classes
    )
    model_cfg = model.pretrained_cfg
    print(f"model_cfg: {model_cfg}")
    datamodule = ClassificationDataModule(args, model_cfg)
    criterion = nn.CrossEntropyLoss()
    noqat_exp_name = args.exp_name.replace("_qat", "")
    best_ckpt_path = os.path.join(
        args.weight_root, f"{noqat_exp_name}_{args.model_name}_best_local.ckpt"
    )
    plmodel = ImageClassifier.load_from_checkpoint(
        checkpoint_path=best_ckpt_path,
        args=args,
        model=model,
        criterion=criterion,
    )
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        mode="min",
        dirpath=to_absolute_path(args.weight_root),
        filename=f"{args.exp_name}_{args.model_name}_best_qat",
    )
    callbacks = [
        TQDMProgressBar(args.print_freq),
        QuantizationAwareTraining(),
        checkpoint_callback,
    ]
    trainer = pl.Trainer(
        logger=mlf_logger,
        accelerator="gpu",
        devices=1,
        max_epochs=args.epochs,
        log_every_n_steps=args.log_freq,
        strategy="dp",
        precision=16 if args.apex else 32,
        deterministic=True,
        num_sanity_val_steps=0,
        callbacks=callbacks,
    )
    timer = ElapsedTimePrinter()
    timer.start()
    trainer.fit(plmodel, datamodule=datamodule)
    timer.end()
    convert_script_model(args, model)


if __name__ == "__main__":
    main()
