import hydra
import pytorch_lightning as L
import segmentation_models_pytorch as smp
from hydra.utils import to_absolute_path
from ImageSegmentator import ImageSegmentator
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from pytorch_lightning.loggers import MLFlowLogger
from segmentation_models_pytorch import utils as smp_utils
from utils import ElapsedTimePrinter
from VOCSegDataModule import VOCSegDataModule


@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(args: DictConfig) -> None:
    print(OmegaConf.to_yaml(args))
    L.seed_everything(args.seed)

    mlf_logger = MLFlowLogger(
        experiment_name="5_segmentation_voc",
        log_model=True,
    )
    mlf_logger.log_hyperparams(args)

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
    metrics = smp_utils.metrics.IoU(threshold=args.iou_threshold)
    plmodel = ImageSegmentator(args, model, criterions, criterions_weight, metrics)
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        mode="min",
        dirpath=to_absolute_path(args.weight_root),
        filename=f"{args.exp_name}_{args.arch.encoder}_{args.arch.decoder}_best",
    )
    callbacks = [TQDMProgressBar(args.print_freq), checkpoint_callback]

    trainer = L.Trainer(
        logger=mlf_logger,
        accelerator="gpu",
        devices=1,
        max_epochs=args.epochs,
        log_every_n_steps=args.log_freq,
        strategy="ddp_find_unused_parameters_true",
        precision=16 if args.apex else 32,
        deterministic=False,
        num_sanity_val_steps=0,
        callbacks=callbacks,
    )

    timer = ElapsedTimePrinter()
    timer.start()
    trainer.fit(plmodel, datamodule=datamodule)
    timer.end()
    trainer.test(plmodel, datamodule=datamodule, verbose=True)


if __name__ == "__main__":
    main()
