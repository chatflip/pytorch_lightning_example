import pytorch_lightning as L
import torch.optim as optim
from pytorch_lightning.callbacks import LearningRateMonitor


class ImageSegmentator(L.LightningModule):
    def __init__(self, args, model, criterions, criterions_weight, metrics):
        super().__init__()
        self.validation_step_outputs = []
        self.args = args
        self.model = model
        self.criterions = criterions
        self.criterions_weight = criterions_weight
        self.metrics = metrics

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_nb):
        image, target = batch
        output = self(image)
        loss = 0
        for key in self.criterions.keys():
            single_loss = self.criterions[key](output, target)
            loss += self.criterions_weight[key] * single_loss
            # GPU:0の結果のみlog保存
            self.log(f"train_{key}", single_loss.item())
        iou_acc = self.metrics(output, target)
        self.log("train_loss", loss.item())
        self.log("train_iou", iou_acc.item())
        return {"loss": loss}

    def training_step_end(self, batch_parts):
        return batch_parts["loss"].mean()

    def validation_step(self, batch, batch_idx):
        image, target = batch
        output = self(image)
        result_dict = {}
        loss = 0
        for key in self.criterions.keys():
            single_loss = self.criterions[key](output, target)
            loss += self.criterions_weight[key] * single_loss
            result_dict[key] = loss.item()
        iou_acc = self.metrics(output, target)
        result_dict["loss"] = loss.item()
        result_dict["iou"] = iou_acc.item()
        self.validation_step_outputs.append(result_dict)
        return result_dict

    def on_validation_epoch_end(self):
        # TODO: drop_last分のbatchsizeの考慮されてないから厳密ではない
        for key in self.validation_step_outputs[0].keys():
            # validation_stepのdictのkeyごとに集計
            results = [
                self.validation_step_outputs[i][key]
                for i in range(len(self.validation_step_outputs))
            ]
            self.log(f"val_{key}", float(sum(results) / len(results)))

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def on_test_epoch_end(self):
        self.on_validation_epoch_end()

    def configure_optimizers(self):
        num_training_samples = len(self.trainer.datamodule.train_dataloader())
        optimizer = optim.Adam(
            self.parameters(),
            lr=self.args.lr,
        )  # 最適化方法定義
        lr_scheduler = {
            "scheduler": optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer,
                T_0=5 * num_training_samples,
                T_mult=2,
                eta_min=0.1 * self.args.lr,
                last_epoch=-1,
            ),
            "interval": "step",
            "name": "lr",
        }
        return [optimizer], [lr_scheduler]

    def configure_callbacks(self):
        lr_monitor = LearningRateMonitor(logging_interval="step")
        return [lr_monitor]
