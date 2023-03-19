import os

from omegaconf import DictConfig, OmegaConf
import hydra
from pytorch_lightning.callbacks import ModelCheckpoint

import hydra
import pytorch_lightning as pl
import torch
import torch.nn as nn
from AnimeFaceDataModule import AnimeFaceDataModule
from ImageClassifier import ImageClassifier
from model import mobilenet_v2
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from utils import ElapsedTimePrinter
from pytorch_lightning.loggers import MLFlowLogger
from hydra.utils import  to_absolute_path

def convert_script_model(args, model):
    cwd = hydra.utils.get_original_cwd()
    model_path = os.path.join(cwd, args.weight_root, f"{args.exp_name}_mobilenetv2.pt")
    model.eval()
    script_model = torch.jit.script(model)
    torch.jit.save(script_model, model_path)

@hydra.main(version_base=None, config_path="../config", config_name="base")
def main(args: DictConfig) -> None:
    print(OmegaConf.to_yaml(args))

    mlf_logger = MLFlowLogger(
        experiment_name="1_classification_animeface",
        log_model=True,
    )
    mlf_logger.log_hyperparams(args)

    pl.seed_everything(args.seed)
    model = mobilenet_v2(pretrained=True, num_classes=args.num_classes)
    datamodule = AnimeFaceDataModule(args)
    criterion = nn.CrossEntropyLoss()
    plmodel = ImageClassifier(args, model, criterion)
    
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        mode="min",
        dirpath=to_absolute_path(args.weight_root),
        filename=f"{args.exp_name}_mobilenetv2_best",
    )
    callbacks = [TQDMProgressBar(args.print_freq), checkpoint_callback]
    trainer = pl.Trainer(
        logger=mlf_logger,
        accelerator='gpu',
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
    trainer.test(plmodel, datamodule=datamodule, verbose=True)
    convert_script_model(args, model)



if __name__ == "__main__":
    main()
