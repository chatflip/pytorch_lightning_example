import argparse
import os
import random
import shutil

from pathlib import Path

from loguru import logger

from Food101Downloader import Food101Downloader
from Food101Builder import Food101Builder


def main(args: argparse.Namespace) -> None:
    downloader = Food101Downloader(args.download_dir)
    downloader()
    builder = Food101Builder(args.download_dir, args.dataset_dir)
    builder()

if __name__ == "__main__":
    """メイン関数."""
    parser = argparse.ArgumentParser(
        description="Food101データセットの準備（ダウンロード・分割）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--download_dir",
        type=Path,
        default=Path("data/raw/food-101"),
        help="ダウンロードしたデータを保存するディレクトリ",
    )
    parser.add_argument(
        "--dataset_dir",
        type=Path,
        default=Path("data/datasets/food101"),
        help="分割後のデータを保存するディレクトリ",
    )
    args = parser.parse_args()
    main(args)

