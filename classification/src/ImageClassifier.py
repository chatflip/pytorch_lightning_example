import pytorch_lightning as L
import torch.optim as optim

from utils import accuracy


class ImageClassifier(L.LightningModule):
    def __init__(self, args, model, criterion):
        super().__init__()
        self.validation_step_outputs = []
        self.args = args
        self.model = model
        self.criterion = criterion

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_nb):
        image, target = batch
        output = self(image)
        loss = self.criterion(output, target)
        if self.global_step % self.args.log_freq == 0:
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            # GPU:0の結果のみlog保存
            self.log("train_loss", loss.item())
            self.log("train_acc1", acc1.item())
            self.log("train_acc5", acc5.item())
        return {"loss": loss}

    def training_step_end(self, batch_parts):
        return batch_parts["loss"].mean()

    def validation_step(self, batch, batch_idx):
        image, target = batch
        output = self(image)
        loss = self.criterion(output, target)
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        self.validation_step_outputs.append([loss, acc1, acc5])
        return loss.item(), acc1.item(), acc5.item()

    def on_validation_epoch_end(self):
        # TODO: drop_lastの場合計算合わない
        loss_list = []
        acc1_list = []
        acc5_list = []
        for output in self.validation_step_outputs:
            loss_list.append(output[0].cpu().numpy())
            acc1_list.append(output[1].cpu().numpy())
            acc5_list.append(output[2].cpu().numpy())
        loss = sum(loss_list) / len(loss_list)
        acc1 = sum(acc1_list) / len(acc1_list)
        acc5 = sum(acc5_list) / len(acc5_list)
        self.log("val_loss", float(loss))
        self.log("val_acc1", float(acc1))
        self.log("val_acc5", float(acc5))

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def on_test_epoch_end(self):
        self.on_validation_epoch_end()

    def configure_optimizers(self):
        num_train_sample = len(self.trainer.datamodule.train_dataloader())
        optimizer = optim.SGD(
            self.parameters(),
            lr=self.args.optimizer.lr,
            momentum=self.args.optimizer.momentum,
            weight_decay=self.args.optimizer.weight_decay,
        )  # 最適化方法定義
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=self.args.optimizer.lr_step_size * num_train_sample,
            gamma=self.args.optimizer.lr_gamma,
        )
        return [optimizer], [scheduler]
