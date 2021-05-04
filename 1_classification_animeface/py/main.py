import os
import time

import hydra
import pytorch_lightning as pl
import torch.nn as nn
from AnimeFaceDataModule import AnimeFaceDataModule
from CustomMlFlowLogger import CustomMlFlowLogger
from ImageClassifier import ImageClassifier
from MlflowWriter import MlflowWriter
from model import mobilenet_v2


def write_log_base(args, writer):
    for key in args:
        writer.log_param(key, args[key])
    writer.log_params_from_omegaconf_dict(args)
    writer.log_artifact(os.path.join(os.getcwd(), ".hydra/config.yaml"))
    writer.log_artifact(os.path.join(os.getcwd(), ".hydra/hydra.yaml"))
    writer.log_artifact(os.path.join(os.getcwd(), ".hydra/overrides.yaml"))
    return writer


@hydra.main(config_name="./../config/config.yaml")
def main(args):
    writer = MlflowWriter(args.exp_name)
    writer = write_log_base(args, writer)
    logger = CustomMlFlowLogger(writer)

    pl.seed_everything(args.seed)
    model = mobilenet_v2(pretrained=True, num_classes=args.num_classes)
    datamodule = AnimeFaceDataModule(args)
    criterion = nn.CrossEntropyLoss()
    plmodel = ImageClassifier(args, model, criterion)
    trainer = pl.Trainer(
        logger=logger,
        checkpoint_callback=False,
        gpus=2,
        max_epochs=args.epochs,
        flush_logs_every_n_steps=args.print_freq,
        log_every_n_steps=args.log_freq,
        accelerator="dp",
        precision=16 if args.apex else 32,
        deterministic=True,
        num_sanity_val_steps=-1,
    )
    starttime = time.time()  # 実行時間計測(実時間)
    trainer.fit(plmodel, datamodule=datamodule)
    trainer.test(plmodel, datamodule=datamodule, verbose=True)
    writer.move_mlruns()
    # 実行時間表示
    endtime = time.time()
    interval = endtime - starttime
    print(
        "elapsed time = {0:d}h {1:d}m {2:d}s".format(
            int(interval / 3600),
            int((interval % 3600) / 60),
            int((interval % 3600) % 60),
        )
    )


if __name__ == "__main__":
    main()
