import os
import sys
import tarfile
import urllib.error
import urllib.request

import cv2
import hydra
import numpy as np
import pandas as pd


class VOC2012Downloader:
    """Downloader for VOC2012 segmentation dataset.

    This class handles downloading, decompressing, and preprocessing
    the VOC2012 dataset for segmentation tasks.
    """

    _BASE_URL = "http://host.robots.ox.ac.uk/pascal/VOC/voc2012"
    _COMPRESSED_FILE = "VOCtrainval_11-May-2012.tar"

    def __init__(self, root_dir: str = "./../datasets") -> None:
        """Initialize VOC2012Downloader.

        Args:
            root_dir: Root directory to save the dataset.
        """
        try:
            cwd = hydra.utils.get_original_cwd()
            self.root_dir = os.path.join(cwd, root_dir)
        except ValueError:
            self.root_dir = root_dir
        self.voc12_root = os.path.join(self.root_dir, "VOCdevkit/VOC2012")
        if not os.path.exists(self.voc12_root):
            self.download()
            self.decompress()
            self.make_raw_annotation()

    def download(self) -> None:
        """ターゲットURLからVOC2012データセットをダウンロードする"""
        target_url = os.path.join(self._BASE_URL, self._COMPRESSED_FILE)
        fullpath = os.path.join(self.root_dir, self._COMPRESSED_FILE)
        if os.path.exists(fullpath):
            print(f"File exists: {target_url}")
        else:
            print(f"Downloading: {target_url}")
            try:
                urllib.request.urlretrieve(
                    url=target_url,
                    filename=fullpath,
                    reporthook=self.progress,
                )
                print("")
            except (OSError, urllib.error.HTTPError) as err:
                print("ERROR :{}".format(err.code))
                print(err.reason)

    def decompress(self) -> None:
        """ダウンロードしたtarファイルを解凍する"""
        filepath = os.path.join(self.root_dir, self._COMPRESSED_FILE)
        filepath = os.path.abspath(filepath)
        with tarfile.open(filepath, "r:*") as tr:
            tr.extractall(path=os.path.join(self.root_dir, ""))
        os.remove(filepath)

    def make_raw_annotation(self) -> None:
        """カラーコード化されたセグメンテーションマスクを生のクラスインデックスに変換する

        このメソッドはカラーコード化されたセグメンテーション画像を読み込み、
        クラスインデックスを含む単一チャネル画像に変換する。
        """
        print("make_raw_annotation")
        color_map = self.get_pascal_labels()
        num_class = color_map.shape[0]
        dst_root = os.path.join(self.voc12_root, "SegmentationRaw")
        os.makedirs(dst_root, exist_ok=True)
        image_list_path = os.path.join(
            self.voc12_root, "ImageSets", "Segmentation", "trainval.txt"
        )
        image_lists = pd.read_table(image_list_path, header=None)
        for idx, image_list in image_lists.iterrows():
            if idx % 100 == 0:
                print(f"{idx:04d} / {len(image_lists):04d}: {image_list.values[0]}")
            src_path = f"{self.voc12_root}/SegmentationClass/{image_list.values[0]}.png"
            dst_path = f"{self.voc12_root}/SegmentationRaw/{image_list.values[0]}.png"
            annotation_color = cv2.imread(src_path)
            raw_annotation = np.zeros(annotation_color.shape[:2])
            for class_idx in range(num_class):
                valid = np.all(annotation_color == color_map[class_idx], axis=-1)
                rs, cs = valid.nonzero()
                raw_annotation[rs, cs] = class_idx
            raw_annotation = np.array(raw_annotation, dtype=np.uint8)
            cv2.imwrite(dst_path, raw_annotation)

    @staticmethod
    def progress(block_count: int, block_size: int, total_size: int) -> None:
        """ダウンロード進捗コールバック

        Args:
            block_count: これまでにダウンロードしたブロック数。
            block_size: 各ブロックのサイズ（バイト）。
            total_size: ファイルの合計サイズ（バイト）。
        """
        percentage = min(int(100.0 * block_count * block_size / total_size), 100)
        bar = "[{}>{}]".format("=" * (percentage // 4), " " * (25 - percentage // 4))
        sys.stdout.write("{} {:3d}%\r".format(bar, percentage))
        sys.stdout.flush()

    @staticmethod
    def get_pascal_labels() -> np.ndarray:
        """セグメンテーションクラス用のPASCAL VOCカラーマップを取得する

        Returns:
            各クラスのBGR色値を含む形状(num_classes, 3)のNumPy配列。
        """
        VOC_COLOR_MAP: list[list[int]] = [
            [0, 0, 0],
            [128, 0, 0],
            [0, 128, 0],
            [128, 128, 0],
            [0, 0, 128],
            [128, 0, 128],
            [0, 128, 128],
            [128, 128, 128],
            [64, 0, 0],
            [192, 0, 0],
            [64, 128, 0],
            [192, 128, 0],
            [64, 0, 128],
            [192, 0, 128],
            [64, 128, 128],
            [192, 128, 128],
            [0, 64, 0],
            [128, 64, 0],
            [0, 192, 0],
            [128, 192, 0],
            [0, 64, 128],
        ]
        color_map = np.array(VOC_COLOR_MAP, dtype=np.uint8)
        color_map = color_map[:, ::-1]  # RGB2BGR
        return color_map


if __name__ == "__main__":
    VOC2012Downloader()
