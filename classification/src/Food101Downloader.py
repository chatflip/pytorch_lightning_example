import os
import tarfile
import urllib.error
import urllib.request
from pathlib import Path

from loguru import logger


class Food101Downloader:
    """Food101ダウンローダー"""

    _TARGET_URL = "http://data.vision.ee.ethz.ch/cvl/food-101.tar.gz"

    def __init__(
        self, root_dir: Path = Path("datasets"), remove_compressed: bool = False
    ) -> None:
        """Food101ダウンローダーを初期化する

        Args:
            root_dir: データセットを保存するルートディレクトリ。
            remove_compressed: 解凍後に圧縮ファイルを削除するかどうか。
        """
        self.root_dir = root_dir
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self.remove_compressed = remove_compressed

    def download(self) -> None:
        """Food101データセットをダウンロードする

        データセットがまだ存在しない場合はダウンロードし、必要に応じて解凍する。
        オプションで解凍後に圧縮ファイルを削除する。
        """
        compressed_path = self.root_dir / "food-101.tar.gz"
        decompressed_path = self.root_dir / "food-101"
        if not compressed_path.exists():
            self._download_food101(compressed_path)
        else:
            logger.warning(f"Food101 already downloaded: {compressed_path}")

        if not decompressed_path.exists():
            self._decompress_tarfile(compressed_path, self.root_dir)
        else:
            logger.warning(f"Food101 already decompressed: {decompressed_path}")

        if self.remove_compressed:
            os.remove(compressed_path)

    def _download_food101(self, filename: Path) -> None:
        """ターゲットURLからFood101データセットをダウンロードする

        Args:
            filename: ダウンロードしたファイルを保存するパス。
        """
        logger.info(f"Downloading: {self._TARGET_URL}")
        try:
            urllib.request.urlretrieve(
                url=self._TARGET_URL,
                filename=filename.as_posix(),
                reporthook=self._progress_callback,
            )
        except (OSError, urllib.error.HTTPError) as err:
            logger.error(f"Failed to download Food101: {err}")

    def _decompress_tarfile(self, compressed_path: Path, extract_to: Path) -> None:
        """tarファイルを解凍する

        Args:
            compressed_path: 圧縮ファイルのパス。
            extract_to: tarアーカイブを展開するディレクトリ。
        """
        with tarfile.open(compressed_path, "r:gz") as tr:
            tr.extractall(path=extract_to, filter="data")

    @staticmethod
    def _progress_callback(block_count: int, block_size: int, total_size: int) -> None:
        """ダウンロード進捗コールバック

        Args:
            block_count: これまでにダウンロードしたブロック数。
            block_size: 各ブロックのサイズ（バイト）。
            total_size: ファイルの合計サイズ（バイト）。
        """
        percentage = min(int(100.0 * block_count * block_size / total_size), 100)
        bar = "[{}>{}]".format("=" * (percentage // 4), " " * (25 - percentage // 4))
        logger.opt(raw=True).info("{} {:3d}%\r", bar, percentage)


if __name__ == "__main__":
    Food101Downloader().download()
