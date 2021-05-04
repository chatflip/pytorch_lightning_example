import os
import time

import hydra
import pytorch_lightning as pl
import segmentation_models_pytorch as smp
from CustomMlFlowLogger import CustomMlFlowLogger
from ImageSegmentator import ImageSegmentator
from MlflowWriter import MlflowWriter
from VOCSegDataModule import VOCSegDataModule


def write_base_log(args, writer):
    for key in args:
        writer.log_param(key, args[key])
    writer.log_params_from_omegaconf_dict(args)
    writer.log_artifact(os.path.join(os.getcwd(), ".hydra/config.yaml"))
    writer.log_artifact(os.path.join(os.getcwd(), ".hydra/hydra.yaml"))
    writer.log_artifact(os.path.join(os.getcwd(), ".hydra/overrides.yaml"))
    return writer


@hydra.main(config_path="./../config", config_name="config")
def main(args):
    print(args)
    writer = MlflowWriter(args.exp_name)
    writer = write_base_log(args, writer)
    logger = CustomMlFlowLogger(writer)
    pl.seed_everything(args.seed)

    model = getattr(smp, args.arch.decoder)(
        encoder_name=args.arch.encoder,
        encoder_weights="imagenet",
        classes=args.num_classes,
    )
    datamodule = VOCSegDataModule(args)
    criterion = smp.utils.losses.DiceLoss()
    metrics = smp.utils.metrics.IoU(threshold=args.iou_threshold)
    plmodel = ImageSegmentator(args, model, criterion, metrics)
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
