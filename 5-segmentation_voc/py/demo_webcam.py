import sys
import time

import albumentations as A
import cv2
import hydra
import numpy as np
import segmentation_models_pytorch as smp
import torch
from albumentations.pytorch import ToTensorV2
from ImageSegmentator import ImageSegmentator


def get_pascal_labels():
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


def get_transform(args):
    normalize = A.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0
    )
    transform = A.Compose(
        [
            A.Resize(args.arch.image_height, args.arch.image_width),
            normalize,
            ToTensorV2(),
        ]
    )
    return transform


def preprocess_image(image, transform):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    tensor = transform(image=image)["image"]
    return tensor.unsqueeze(0)


def decode_result(result):
    result = result.squeeze(0)
    index = result.argmax(0)
    return index.cpu().numpy()


def make_overlay(frame, seg_image):
    height, width = frame.shape[:2]
    frame = cv2.resize(frame, seg_image.shape[:2], cv2.INTER_AREA)
    overlay_image = cv2.addWeighted(frame, 0.4, seg_image, 0.6, 0.8)
    overlay_image = cv2.resize(overlay_image, (width, height), cv2.INTER_CUBIC)
    return overlay_image


@hydra.main(config_path="./../config", config_name="config")
def main(args):
    print(args)
    cwd = hydra.utils.get_original_cwd()
    model = getattr(smp, args.arch.decoder)(
        encoder_name=args.arch.encoder,
        encoder_weights="imagenet",
        classes=args.num_classes,
    )

    weight_name = "{}/{}/{}_{}_{}_H{}_W{}-v1.ckpt".format(
        cwd,
        args.path2weight,
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
        metrics=None,
    )

    color_map = get_pascal_labels()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # cpuとgpu自動選択
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
        with torch.no_grad():
            result = model(tensor)
        inference_time = time.perf_counter() - start_time

        start_time = time.perf_counter()
        index = decode_result(result)
        postprocess_time = time.perf_counter() - start_time

        start_time = time.perf_counter()
        seg_image = color_map[index]
        overlay = make_overlay(frame, seg_image)
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
