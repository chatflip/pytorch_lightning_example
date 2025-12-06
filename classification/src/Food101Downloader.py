import os
import tarfile
import urllib.error
import urllib.request
from pathlib import Path

from loguru import logger


class Food101Downloader:
    """Food101 downloader."""

    _TARGET_URL = "http://data.vision.ee.ethz.ch/cvl/food-101.tar.gz"

    def __init__(
        self, root_dir: Path = Path("datasets"), remove_compressed: bool = False
    ) -> None:
        """Initialize Food101 downloader.

        Args:
            root_dir (Path): Root directory to save the dataset
            remove_compressed (bool): Remove compressed file after decompression
        """
        self.root_dir = root_dir
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self.remove_compressed = remove_compressed

    def download(self) -> None:
        """Download Food101 dataset.

        Args:
            root_dir (Path): Root directory to save the dataset
            remove_compressed (bool): Remove compressed file after decompression
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
        """Decompress tarfile.

        Args:
            compressed_path (Path): Compressed path
            extract_to (Path): Directory to extract tar archive to
        """
        with tarfile.open(compressed_path, "r:gz") as tr:
            tr.extractall(path=extract_to, filter="data")

    @staticmethod
    def _progress_callback(block_count: int, block_size: int, total_size: int) -> None:
        """Download progress callback.

        Args:
            block_count (int): Block count
            block_size (int): Block size
            total_size (int): Total size
        """
        percentage = min(int(100.0 * block_count * block_size / total_size), 100)
        bar = "[{}>{}]".format("=" * (percentage // 4), " " * (25 - percentage // 4))
        logger.opt(raw=True).info("{} {:3d}%\r", bar, percentage)


if __name__ == "__main__":
    Food101Downloader().download()
