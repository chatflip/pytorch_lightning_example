import json
import os

from PIL import Image
from torch.utils.data import Dataset


class Food101Dataset(Dataset):
    # 初期化
    def __init__(self, root, phase, transform=None):
        self.transform = transform  # 画像変形用
        self.image_paths = []  # 画像のパス格納用
        self.image_labels = []  # 画像のラベル格納用
        image_root = os.path.join(root, "images")
        metadata_root = os.path.join(root, "meta")
        if phase == "train":
            image_dict_path = os.path.join(metadata_root, "train.json")
        else:
            image_dict_path = os.path.join(metadata_root, "test.json")
        class_name_path = os.path.join(metadata_root, "classes.txt")
        with open(class_name_path, newline=None, mode="r") as f:
            class_names = [s.strip() for s in f.readlines()]

        with open(image_dict_path) as f:
            image_dict = json.load(f)
        for i, class_name in enumerate(class_names):
            filenames = image_dict[class_name]
            image_paths = [os.path.join(image_root, f"{f}.jpg") for f in filenames]
            self.image_labels.extend([i] * len(filenames))
            self.image_paths.extend(image_paths)

    # num_worker数で並列処理される関数
    def __getitem__(self, index):
        image_path = self.image_paths[index]
        image = Image.open(image_path).convert("RGB")  # 画像をPILで開く
        if self.transform is not None:
            image = self.transform(image)  # 画像変形適用
        return image, self.image_labels[index]  # 画像とラベルを返す

    # データセットの画像数宣言(これが無いとエラー)
    def __len__(self):
        return len(self.image_paths)
