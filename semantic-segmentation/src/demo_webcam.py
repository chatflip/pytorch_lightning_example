import sys
import time

import albumentations as A
import cv2
import hydra
import numpy as np
import segmentation_models_pytorch as smp
import torch
from albumentations.pytorch import ToTensorV2
from omegaconf import DictConfig, OmegaConf

from .ImageSegmentator import ImageSegmentator


def get_pascal_labels() -> np.ndarray:
    """セグメンテーションクラス用のPASCAL VOCカラーマップを取得する

    Returns:
        各クラスのBGR色値を含む形状(num_classes, 3)のNumPy配列。
    """
    VOC_COLOR_MAP = [
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


def get_transform(args: DictConfig) -> A.Compose:
    """推論用の画像変換パイプラインを取得する

    Args:
        args: arch設定を含む設定辞書。

    Returns:
        リサイズ、正規化、テンソル変換を含むAlbumentations Composeオブジェクト。
    """
    transform = A.Compose(
        [
            A.Resize(args.arch.image_height, args.arch.image_width),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ]
    )
    return transform


def preprocess_image(
    image: np.ndarray, transform: A.Compose
) -> torch.Tensor:
    """モデル推論用に画像を前処理する

    Args:
        image: BGR形式の入力画像。
        transform: Albumentations変換パイプライン。

    Returns:
        バッチ次元を含む前処理済み画像テンソル。
    """
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    tensor = transform(image=image)["image"]
    return tensor.unsqueeze(0)


def decode_result(result: torch.Tensor) -> np.ndarray:
    """モデル出力をクラスインデックスにデコードする

    Args:
        result: 形状(batch, num_classes, height, width)のモデル出力テンソル。

    Returns:
        形状(height, width)のクラスインデックスのNumPy配列。
    """
    index = result.argmax(0).squeeze(0)
    return index.cpu().numpy()


def make_overlay(
    frame: np.ndarray, color_map: np.ndarray, index: np.ndarray
) -> np.ndarray:
    """元のフレーム上にセグメンテーションのオーバーレイ可視化を作成する

    Args:
        frame: 元のビデオフレーム。
        color_map: 可視化用のカラーマップ配列。
        index: セグメンテーション結果からのクラスインデックス配列。

    Returns:
        元のフレームとセグメンテーション色を組み合わせたオーバーレイ画像。
    """
    height, width = frame.shape[:2]
    seg_image = color_map[index]
    seg_image = cv2.resize(frame, (width, height), cv2.INTER_AREA)
    overlay_image = cv2.addWeighted(frame, 0.4, seg_image, 0.6, 0.8)
    return overlay_image


@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(args: DictConfig) -> None:
    """リアルタイムウェブカメラセグメンテーションデモ用のメイン関数

    Args:
        args: モデル設定、アーキテクチャパラメータなどを含む
            Hydraからの設定辞書。
    """
    print(OmegaConf.to_yaml(args))
    cwd = hydra.utils.get_original_cwd()
    model = getattr(smp, args.arch.decoder)(
        encoder_name=args.arch.encoder,
        encoder_weights="imagenet",
        classes=args.num_classes,
    )

    weight_name = "{}/{}/{}_{}_{}_H{}_W{}.ckpt".format(
        cwd,
        args.weight_root,
        args.exp_name,
        args.arch.decoder,
        args.arch.encoder,
        args.arch.image_height,
        args.arch.image_width,
    )

    ImageSegmentator.load_from_checkpoint(
        weight_name,
        args=args,
        model=model,
        criterions=None,
        criterions_weight=None,
        metrics=None,
    )

    color_map = get_pascal_labels()
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    model = model.eval().to(device)

    camera_width = 1280
    camera_height = 720
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, camera_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, camera_height)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FPS, 60.0)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    print(f"video FPS: {cap.get(cv2.CAP_PROP_FPS)}")
    transform = get_transform(args)

    while cap.isOpened():
        start_time = time.perf_counter()
        ret, frame = cap.read()

        tensor = preprocess_image(frame, transform)
        tensor = tensor.to(device)
        preprocess_time = time.perf_counter() - start_time

        start_time = time.perf_counter()
        with torch.inference_mode():
            result = model(tensor)
        inference_time = time.perf_counter() - start_time

        start_time = time.perf_counter()
        index = decode_result(result)
        postprocess_time = time.perf_counter() - start_time

        start_time = time.perf_counter()
        overlay = make_overlay(frame, color_map, index)
        visualize_time = time.perf_counter() - start_time

        cv2.imshow("result", overlay)
        interval = preprocess_time + inference_time + postprocess_time + visualize_time
        sys.stdout.write("\rFPS: {:.1f}".format(1.0 / interval))
        sys.stdout.flush()
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()


if __name__ == "__main__":
    main()
