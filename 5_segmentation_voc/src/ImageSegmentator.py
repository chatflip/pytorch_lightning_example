import pytorch_lightning as L
import torch
import torch.nn as nn
import torch.optim as optim
from omegaconf import DictConfig
from pytorch_lightning.callbacks import Callback, LearningRateMonitor
from torch.optim import Optimizer

from segmentation_models_pytorch import utils as smp_utils


class ImageSegmentator(L.LightningModule):
    """画像セグメンテーション用のPyTorch Lightningモジュール

    このモジュールは複数の損失関数とメトリクスを使用して、
    画像セグメンテーションタスクの訓練、検証、テストを処理する。
    """

    def __init__(
        self,
        args: DictConfig,
        model: nn.Module,
        criterions: dict[str, nn.Module],
        criterions_weight: dict[str, float],
        metrics: smp_utils.metrics.IoU,
    ) -> None:
        """ImageSegmentatorを初期化する

        Args:
            args: lrなどを含む設定引数。
            model: 訓練するニューラルネットワークモデル。
            criterions: 損失関数の辞書。
            criterions_weight: 各損失関数の重みの辞書。
            metrics: メトリクス関数（例: IoU）。
        """
        super().__init__()
        self.validation_step_outputs: list[dict[str, float]] = []
        self.args = args
        self.model = model
        self.criterions = criterions
        self.criterions_weight = criterions_weight
        self.metrics = metrics

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """モデルを通した順伝播

        Args:
            x: 入力テンソル。

        Returns:
            モデルの出力テンソル。
        """
        return self.model(x)

    def training_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_nb: int
    ) -> dict[str, torch.Tensor]:
        """訓練ステップ

        Args:
            batch: (画像, ターゲット)のタプル。
            batch_nb: バッチ番号。

        Returns:
            損失を含む辞書。
        """
        image, target = batch
        output = self(image)
        loss = 0.0
        for key in self.criterions.keys():
            single_loss = self.criterions[key](output, target)
            loss += self.criterions_weight[key] * single_loss
            # GPU:0の結果のみlog保存
            self.log(f"train_{key}", single_loss.item())
        iou_acc = self.metrics(output, target)
        self.log("train_loss", loss.item())
        self.log("train_iou", iou_acc.item())
        return {"loss": loss}

    def training_step_end(
        self, batch_parts: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """複数のGPU間で訓練ステップの出力を集約する

        Args:
            batch_parts: 各GPUからの損失を含む辞書。

        Returns:
            すべてのGPU間の平均損失。
        """
        return batch_parts["loss"].mean()

    def validation_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> dict[str, float]:
        """検証ステップ

        Args:
            batch: (画像, ターゲット)のタプル。
            batch_idx: バッチインデックス。

        Returns:
            各基準の損失とメトリクスを含む辞書。
        """
        image, target = batch
        output = self(image)
        result_dict: dict[str, float] = {}
        loss = 0.0
        for key in self.criterions.keys():
            single_loss = self.criterions[key](output, target)
            loss += self.criterions_weight[key] * single_loss
            result_dict[key] = single_loss.item()
        iou_acc = self.metrics(output, target)
        result_dict["loss"] = loss.item()
        result_dict["iou"] = iou_acc.item()
        self.validation_step_outputs.append(result_dict)
        return result_dict

    def on_validation_epoch_end(self) -> None:
        """検証エポック終了時に呼び出される

        検証メトリクスを集約してログに記録する。
        注意: drop_last分のbatchsizeが考慮されていないため、厳密ではない。
        """
        if len(self.validation_step_outputs) == 0:
            return
        for key in self.validation_step_outputs[0].keys():
            # validation_stepのdictのkeyごとに集計
            results = [
                self.validation_step_outputs[i][key]
                for i in range(len(self.validation_step_outputs))
            ]
            self.log(f"val_{key}", float(sum(results) / len(results)))
        self.validation_step_outputs.clear()

    def test_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> dict[str, float]:
        """テストステップ

        Args:
            batch: (画像, ターゲット)のタプル。
            batch_idx: バッチインデックス。

        Returns:
            各基準の損失とメトリクスを含む辞書。
        """
        return self.validation_step(batch, batch_idx)

    def on_test_epoch_end(self) -> None:
        """テストエポック終了時に呼び出される

        テストメトリクスを集約してログに記録する。
        """
        self.on_validation_epoch_end()

    def configure_optimizers(
        self,
    ) -> tuple[list[Optimizer], list[dict[str, optim.lr_scheduler.LRScheduler | str | int]]]:
        """オプティマイザーと学習率スケジューラーを設定する

        Returns:
            (オプティマイザーのリスト, スケジューラー辞書のリスト)のタプル。
        """
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

    def configure_callbacks(self) -> list[Callback]:
        """訓練用のコールバックを設定する

        Returns:
            コールバックのリスト。
        """
        lr_monitor = LearningRateMonitor(logging_interval="step")
        return [lr_monitor]
