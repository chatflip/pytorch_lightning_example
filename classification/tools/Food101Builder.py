from pathlib import Path
from loguru import logger
import shutil
class Food101Builder:
    """Food101データセットを構築するクラス."""

    def __init__(self, download_dir: Path, dataset_dir: Path) -> None:
        self.download_dir = download_dir
        self.meta_dir = download_dir / "meta"
        self.train_dir = dataset_dir / "train"
        self.val_dir = dataset_dir / "val"

        
    def __call__(self) -> None:
        """Food101データセットを構築する."""
        logger.info("Building train and val datasets...")

        train_meta_path = self.meta_dir / "train.txt"
        with open(train_meta_path, "r") as f:
            train_lines = f.readlines()
        train_paths = [line.strip() for line in train_lines]
        logger.info(f"Found {len(train_paths)} train images")

        val_meta_path = self.meta_dir / "test.txt"
        with open(val_meta_path, "r") as f:
            val_lines = f.readlines()
        val_paths = [line.strip() for line in val_lines]
        logger.info(f"Found {len(val_paths)} val images")
        
        for train_path in train_paths:
            src_image_path = self.download_dir / "images" / f"{train_path}.jpg"
            dst_image_path = self.train_dir / f"{train_path}.jpg"
            dst_image_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(src_image_path, dst_image_path)
        for val_path in val_paths:
            src_image_path = self.download_dir / "images" / f"{val_path}.jpg"
            dst_image_path = self.val_dir / f"{val_path}.jpg"
            dst_image_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(src_image_path, dst_image_path)
        
        logger.info("Food101 Dataset Building Complete")