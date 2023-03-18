import os

import hydra
import pytorch_lightning as pl
import torch
import torch.nn as nn
from AnimeFaceDataModule import AnimeFaceDataModule
from CustomMlFlowLogger import CustomMlFlowLogger
from ImageClassifier import ImageClassifier
from MlflowWriter import MlflowWriter
from model import mobilenet_v2
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from utils import ElapsedTimePrinter


def convert_script_model(args, model):
    cwd = hydra.utils.get_original_cwd()
    model_path = os.path.join(cwd, args.weight_root, f"{args.exp_name}_mobilenetv2.pt")
    model.eval()
    script_model = torch.jit.script(model)
    torch.jit.save(script_model, model_path)


@hydra.main(config_path="./../config", config_name="base.yaml")
def main(args):
    print(args)
    timer = ElapsedTimePrinter()
    writer = MlflowWriter(args.exp_name)
    writer.write_hydra_args(args)
    logger = CustomMlFlowLogger(writer)

    pl.seed_everything(args.seed)
    model = mobilenet_v2(pretrained=True, num_classes=args.num_classes)
    datamodule = AnimeFaceDataModule(args)
    criterion = nn.CrossEntropyLoss()
    plmodel = ImageClassifier(args, model, criterion)
    callbacks = [TQDMProgressBar(args.print_freq)]
    trainer = pl.Trainer(
        logger=logger,
        enable_checkpointing=False,
        gpus=1,
        max_epochs=args.epochs,
        log_every_n_steps=args.log_freq,
        strategy="dp",
        precision=16 if args.apex else 32,
        deterministic=True,
        num_sanity_val_steps=0,
        callbacks=callbacks,
    )
    timer.start()
    trainer.fit(plmodel, datamodule=datamodule)
    trainer.test(plmodel, datamodule=datamodule, verbose=True)
    writer.move_mlruns()

    convert_script_model(args, model)
    timer.end()


if __name__ == "__main__":
    main()
