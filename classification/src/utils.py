import time

import torch


class AverageMeter:
    """平均値と現在の値を計算して保存する

    このクラスは訓練中のメトリクス記録に使用される。
    """

    def __init__(self, name: str, fmt: str = ":f") -> None:
        """AverageMeterを初期化する

        Args:
            name: メトリクスの名前。
            fmt: 値を表示するためのフォーマット文字列。
        """
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self) -> None:
        """すべての統計をゼロにリセットする"""
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1) -> None:
        """新しい値で統計を更新する

        Args:
            val: 追加する新しい値。
            n: この値が表すサンプル数。
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self) -> str:
        """メーターの文字列表現を返す

        Returns:
            現在の値と平均を含むフォーマット済み文字列。
        """
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)


class ProgressMeter:
    """複数のメトリクスで訓練中の進捗を表示する"""

    def __init__(
        self, num_batches: int, meters: list[AverageMeter], prefix: str = ""
    ) -> None:
        """ProgressMeterを初期化する

        Args:
            num_batches: バッチの総数。
            meters: 表示するAverageMeterインスタンスのリスト。
            prefix: バッチ番号の前に追加するプレフィックス文字列。
        """
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch: int) -> None:
        """現在のバッチの進捗を表示する

        Args:
            batch: 現在のバッチ番号。
        """
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print("\t".join(entries))

    def _get_batch_fmtstr(self, num_batches: int) -> str:
        """バッチ番号のフォーマット文字列を生成する

        Args:
            num_batches: バッチの総数。

        Returns:
            バッチ表示用のフォーマット文字列。
        """
        num_digits = len(str(num_batches // 1))
        fmt = "{:" + str(num_digits) + "d}"
        return "[" + fmt + "/" + fmt.format(num_batches) + "]"


def accuracy(
    output: torch.Tensor, target: torch.Tensor, topk: tuple[int, ...] = (1,)
) -> list[torch.Tensor]:
    """指定されたk値について、上位k個の予測に対する精度を計算する

    Args:
        output: 形状(batch_size, num_classes)のモデル出力テンソル。
        target: 形状(batch_size,)の正解ラベル。
        topk: top-k精度を計算するk値のタプル。

    Returns:
        topk内の各kに対する精度テンソルのリスト。
    """
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res: list[torch.Tensor] = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


class ElapsedTimePrinter:
    """経過時間を測定して表示するユーティリティクラス"""

    def __init__(self) -> None:
        """ElapsedTimePrinterを初期化する"""
        self.start_time: float = 0.0
        self.elapsed_time: float = 0.0

    def start(self) -> None:
        """タイマーを開始する"""
        self.start_time = time.perf_counter()

    def end(self) -> None:
        """タイマーを停止して経過時間を表示する"""
        self.elapsed_time = time.perf_counter() - self.start_time
        self.print()

    def print(self) -> None:
        """経過時間を時間、分、秒の形式で表示する"""
        print(
            "elapsed time = {0:d}h {1:d}m {2:d}s".format(
                int(self.elapsed_time / 3600),
                int((self.elapsed_time % 3600) / 60),
                int((self.elapsed_time % 3600) % 60),
            )
        )
