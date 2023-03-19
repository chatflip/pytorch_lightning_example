import os
import sys
import tarfile
import urllib.request


class Food101Downloader:
    _BASE_URL = "http://data.vision.ee.ethz.ch/cvl"
    _COMPRESSED_FILE = "food-101.tar.gz"
    _DECOMPRESS_FOLDER = "food-101"

    def __init__(
        self,
        root_dir="./../datasets",
    ):
        self.root_dir = root_dir
        os.makedirs(self.root_dir, exist_ok=True)

        target_url = os.path.join(self._BASE_URL, self._COMPRESSED_FILE)
        compressed_path = os.path.join(self.root_dir, self._COMPRESSED_FILE)
        decompressed_path = os.path.join(self.root_dir, self._DECOMPRESS_FOLDER)

        if os.path.exists(decompressed_path):
            print(f"Food101 already exist: {decompressed_path}")
            return
        if not os.path.exists(compressed_path):
            self.download(compressed_path, target_url)
        self.decompress_tarfile(compressed_path)
        os.remove(compressed_path)

    def download(self, filename, target_url):
        print(f"Downloading: {target_url}")
        try:
            urllib.request.urlretrieve(
                url=target_url,
                filename=filename,
                reporthook=self.progress,
            )
            print("")
        except (OSError, urllib.error.HTTPError) as err:
            print(f"ERROR :{err.code}")
            print(err.reason)

    def decompress_tarfile(self, compressed_path):
        basename = os.path.dirname(compressed_path)
        print("compressed_path", compressed_path)
        with tarfile.open(compressed_path, "r:gz") as tr:
            tr.extractall(path=os.path.join(basename))

    @staticmethod
    def progress(block_count, block_size, total_size):
        percentage = min(int(100.0 * block_count * block_size / total_size), 100)
        bar = "[{}>{}]".format("=" * (percentage // 4), " " * (25 - percentage // 4))
        sys.stdout.write("{} {:3d}%\r".format(bar, percentage))
        sys.stdout.flush()


if __name__ == "__main__":
    Food101Downloader()
