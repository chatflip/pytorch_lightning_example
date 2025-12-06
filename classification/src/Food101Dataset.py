import json
import os

from PIL import Image
from torch.utils.data import Dataset


class Food101Dataset(Dataset):
    def __init__(self, root, phase, transform=None):
        self.transform = transform
        self.image_paths = []
        self.image_labels = []
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

    def __getitem__(self, index: int) -> tuple[Image.Image, int]:
        image_path = self.image_paths[index]
        image = Image.open(image_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, self.image_labels[index]

    def __len__(self) -> int:
        return len(self.image_paths)
