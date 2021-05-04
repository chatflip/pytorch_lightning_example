import fnmatch
import glob
import os
import random
import shutil
import sys
import urllib.request
import zipfile

import hydra
from PIL import Image


class AnimeFaceDownloader:
    _BASE_URL = "http://www.nurs.or.jp/~nagadomi/animeface-character-dataset/data"
    _COMPRESSED_FILE = "animeface-character-dataset.zip"

    def __init__(
        self,
        root_dir="./../datasets/animeface",
        raw_path="data",
        tmp_dir="tmp",
        filename="animeface-character-dataset",
    ):
        cwd = hydra.utils.get_original_cwd()
        self.root_dir = os.path.join(cwd, root_dir)
        self.raw_path = raw_path
        self.tmp_dir = tmp_dir
        self.filename = filename
        if not os.path.exists(os.path.join(self.root_dir, self.raw_path)):
            os.makedirs(self.root_dir, exist_ok=True)
            os.makedirs(os.path.join(self.root_dir, self.raw_path), exist_ok=True)
            self.download()
            self.decompress_file()
            self.train_val_split()

    def download(self):
        target_url = os.path.join(self._BASE_URL, self._COMPRESSED_FILE)
        fullpath = os.path.join(self.root_dir, self._COMPRESSED_FILE)
        print("Downloading: {}".format(target_url))
        try:
            urllib.request.urlretrieve(
                url=target_url, filename=fullpath, reporthook=self.progress
            )
            print("")
        except (OSError, urllib.error.HTTPError) as err:
            print("ERROR :{}".format(err.code))
            print(err.reason)

    def decompress_file(self):
        with zipfile.ZipFile(
            os.path.join(self.root_dir, self._COMPRESSED_FILE), "r"
        ) as z:
            z.extractall(os.path.join(self.root_dir, ""))

    def train_val_split(self):
        src = os.path.join(self.root_dir, self.filename, "thumb")
        dst = os.path.join(self.root_dir, self.raw_path)
        os.makedirs(dst, exist_ok=True)
        os.makedirs(os.path.join(dst, "train"), exist_ok=True)
        os.makedirs(os.path.join(dst, "val"), exist_ok=True)
        train_rate = 0.75
        class_names = os.listdir(src)
        class_names.sort()
        random.seed(1)
        num_classes = 0
        for class_name in class_names:
            file_names = glob.glob(os.path.join(src, class_name, "*.png"))
            img_names = fnmatch.filter(file_names, "*.png")
            class_length = len(img_names)
            if class_length <= 4:
                print("passed ", class_name)
                continue
            else:
                print(class_name)
                random.shuffle(img_names)
                os.makedirs(os.path.join(dst, "train", class_name), exist_ok=True)
                os.makedirs(os.path.join(dst, "val", class_name), exist_ok=True)
                for i, img_name in enumerate(img_names):
                    img = Image.open(img_name)
                    name = os.path.basename(img_name)
                    if i < class_length * train_rate:
                        img.save(os.path.join(dst, "train", class_name, name))
                    else:
                        img.save(os.path.join(dst, "val", class_name, name))
                    img.close()
                num_classes += 1
        print("num of classes: {}".format(num_classes))
        self.teardown()

    def teardown(self):
        shutil.rmtree(os.path.join(self.root_dir, self.filename))

    @staticmethod
    def progress(block_count, block_size, total_size):
        percentage = min(int(100.0 * block_count * block_size / total_size), 100)
        bar = "[{}>{}]".format("=" * (percentage // 4), " " * (25 - percentage // 4))
        sys.stdout.write("{} {:3d}%\r".format(bar, percentage))
        sys.stdout.flush()


if __name__ == "__main__":
    a = AnimeFaceDownloader()
