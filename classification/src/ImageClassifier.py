from typing import Any

import pytorch_lightning as L
import torch
import torch.nn as nn
import torch.optim as optim
from omegaconf import DictConfig

from utils import accuracy


class ImageClassifier(L.LightningModule):
    """画像分類用のPyTorch Lightningモジュール

    このモジュールは画像分類タスクの訓練、検証、テストを処理し、
    ログ記録と最適化設定を行う。
    """

    def __init__(
        self, args: DictConfig, model: nn.Module, criterion: nn.Module
    ) -> None:
        """ImageClassifierを初期化する

        Args:
            args: オプティマイザー設定、log_freqなどを含む設定引数。
            model: 訓練するニューラルネットワークモデル。
            criterion: 訓練に使用する損失関数。
        """
        super().__init__()
        self.validation_step_outputs: list[list[torch.Tensor]] = []
        self.args: DictConfig = args  # type: ignore[assignment]
        self.model = model
        self.criterion = criterion

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
        loss = self.criterion(output, target)
        if self.global_step % self.args.log_freq == 0:
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            # GPU:0の結果のみlog保存
            self.log("train_loss", loss.item())
            self.log("train_acc1", acc1.item())
            self.log("train_acc5", acc5.item())
        return {"loss": loss}

    def training_step_end(self, batch_parts: dict[str, torch.Tensor]) -> torch.Tensor:
        """複数のGPU間で訓練ステップの出力を集約する

        Args:
            batch_parts: 各GPUからの損失を含む辞書。

        Returns:
            すべてのGPU間の平均損失。
        """
        return batch_parts["loss"].mean()

    def validation_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> Any:
        """検証ステップ

        Args:
            batch: (画像, ターゲット)のタプル。
            batch_idx: バッチインデックス。

        Returns:
            (損失, top1精度, top5精度)のタプル。
        """
        image, target = batch
        output = self(image)
        loss = self.criterion(output, target)
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        self.validation_step_outputs.append([loss, acc1, acc5])
        return loss.item(), acc1.item(), acc5.item()

    def on_validation_epoch_end(self) -> None:
        """検証エポック終了時に呼び出される

        検証メトリクスを集約してログに記録する。
        注意: drop_lastの場合、計算が合わない可能性がある。
        """
        loss_list: list[float] = []
        acc1_list: list[float] = []
        acc5_list: list[float] = []
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
        self.validation_step_outputs.clear()

    def test_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> Any:
        """テストステップ

        Args:
            batch: (画像, ターゲット)のタプル。
            batch_idx: バッチインデックス。

        Returns:
            (損失, top1精度, top5精度)のタプル。
        """
        return self.validation_step(batch, batch_idx)

    def on_test_epoch_end(self) -> None:
        """テストエポック終了時に呼び出される

        テストメトリクスを集約してログに記録する。
        """
        self.on_validation_epoch_end()

    def configure_optimizers(
        self,
    ) -> Any:
        """オプティマイザーと学習率スケジューラーを設定する

        Returns:
            (オプティマイザーのリスト, スケジューラー辞書のリスト)のタプル。
        """
        if self.trainer.datamodule is None:  # type: ignore[union-attr]
            raise ValueError("datamodule must be set")
        num_train_sample = len(self.trainer.datamodule.train_dataloader())  # type: ignore[union-attr]
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
        return [optimizer], [{"scheduler": scheduler}]
