import time

import hydra
import pytorch_lightning as pl
import segmentation_models_pytorch as smp
from CustomMlFlowLogger import CustomMlFlowLogger
from ImageSegmentator import ImageSegmentator
from MlflowWriter import MlflowWriter
from VOCSegDataModule import VOCSegDataModule


@hydra.main(config_path="./../config", config_name="config")
def main(args):
    print(args)
    writer = MlflowWriter(args.exp_name)
    writer = writer.write_base_log(writer, args)
    logger = CustomMlFlowLogger(writer)
    pl.seed_everything(args.seed)

    model = getattr(smp, args.arch.decoder)(
        encoder_name=args.arch.encoder,
        encoder_weights="imagenet",
        classes=args.num_classes,
    )
    datamodule = VOCSegDataModule(args)
    criterions = {
        "jaccard_loss": smp.losses.JaccardLoss(mode="binary"),
        "dice_loss": smp.losses.DiceLoss(mode="binary"),
        "lovasz_loss": smp.losses.LovaszLoss(mode="binary"),
        "bce_loss": smp.losses.SoftBCEWithLogitsLoss(),
    }
    criterions_weight = {
        "jaccard_loss": 1 / len(criterions),
        "dice_loss": 1 / len(criterions),
        "lovasz_loss": 1 / len(criterions),
        "bce_loss": 1 / len(criterions),
    }
    metrics = smp.utils.metrics.IoU(threshold=args.iou_threshold)
    plmodel = ImageSegmentator(args, model, criterions, criterions_weight, metrics)
    trainer = pl.Trainer(
        logger=logger,
        checkpoint_callback=False,
        gpus=2,
        max_epochs=args.epochs,
        progress_bar_refresh_rate=args.print_freq,
        log_every_n_steps=args.log_freq,
        accelerator="dp",
        precision=16 if args.apex else 32,
        deterministic=True,
        num_sanity_val_steps=0,
    )

    starttime = time.time()  # 実行時間計測(実時間)

    trainer.fit(plmodel, datamodule=datamodule)
    trainer.test(plmodel, datamodule=datamodule, verbose=True)

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
