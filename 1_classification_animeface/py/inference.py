import os

import torch
from utils import ElapsedTimePrinter

if __name__ == "__main__":
    weight_root = "weight"
    image_height = 244
    image_width = 244
    force_cpu = False
    model_path = os.path.join(weight_root, "animeface_mobilenetv2.pt")
    quant_model_path = os.path.join(weight_root, "animeface_qat_quant_mobilenetv2.pt")

    device = device = torch.device(
        "cuda" if torch.cuda.is_available() and not (force_cpu) else "cpu"
    )
    model = torch.jit.load(model_path).to(device)
    quant_model = torch.jit.load(quant_model_path).to(device)
    dummy_input = torch.rand(1, 3, image_height, image_width).to(device)
    timer = ElapsedTimePrinter()
    num_iteration = 1000
    with torch.inference_mode():
        print("float32")
        timer.start()
        for i in range(num_iteration):
            _ = model(dummy_input)
            torch.cuda.synchronize()
        timer.end()
        print("int8")
        timer.start()
        for i in range(num_iteration):
            _ = quant_model(dummy_input)
            torch.cuda.synchronize()
        timer.end()
