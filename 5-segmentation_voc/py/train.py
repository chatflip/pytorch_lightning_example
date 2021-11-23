import hydra
import pytorch_lightning as pl
import segmentation_models_pytorch as smp
from CustomMlFlowLogger import CustomMlFlowLogger
from ImageSegmentator import ImageSegmentator
from MlflowWriter import MlflowWriter
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from utils import ElapsedTimePrinter
from VOCSegDataModule import VOCSegDataModule


@hydra.main(config_path="./../config", config_name="config")
def main(args):
    print(args)
    timer = ElapsedTimePrinter()
    writer = MlflowWriter(args.exp_name)
    writer.write_hydra_args(args)
    logger = CustomMlFlowLogger(writer)
    pl.seed_everything(args.seed)

    model = getattr(smp, args.arch.decoder)(
        encoder_name=args.arch.encoder,
        encoder_weights="imagenet",
        classes=args.num_classes,
    )
    datamodule = VOCSegDataModule(args)
    criterions = {
        "dice_loss": smp.losses.DiceLoss(mode="binary"),
        "lovasz_loss": smp.losses.LovaszLoss(mode="binary"),
        "focal_loss": smp.losses.FocalLoss(mode="binary"),
    }
    criterions_weight = {
        "dice_loss": 1 / len(criterions),
        "lovasz_loss": 1 / len(criterions),
        "focal_loss": 1 / len(criterions),
    }
    metrics = smp.utils.metrics.IoU(threshold=args.iou_threshold)
    plmodel = ImageSegmentator(args, model, criterions, criterions_weight, metrics)
    callbacks = [TQDMProgressBar(args.print_freq)]
    trainer = pl.Trainer(
        logger=logger,
        enable_checkpointing=False,
        gpus=2,
        max_epochs=args.epochs,
        log_every_n_steps=args.log_freq,
        strategy="dp",
        precision=16 if args.apex else 32,
        deterministic=False,
        num_sanity_val_steps=0,
        callbacks=callbacks,
    )

    timer.start()
    trainer.fit(plmodel, datamodule=datamodule)
    trainer.test(plmodel, datamodule=datamodule, verbose=True)
    writer.move_mlruns()
    timer.end()


if __name__ == "__main__":
    main()
