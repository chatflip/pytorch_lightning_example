import time


class ElapsedTimePrinter:
    """経過時間を測定して表示するユーティリティクラス"""

    def __init__(self) -> None:
        """ElapsedTimePrinterを初期化する"""
        self.start_time: float = 0.0
        self.elapsed_time: float = 0.0

    def start(self) -> None:
        """タイマーを開始する"""
        self.start_time = time.time()

    def end(self) -> None:
        """タイマーを停止して経過時間を表示する"""
        self.elapsed_time = time.time() - self.start_time
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
