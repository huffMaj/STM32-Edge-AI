import torch
from torchvision import models
import torch.nn as nn

MODEL_PATH = "models/gesture_model.pth"
ONNX_PATH = "models/gesture_model.onnx"
NUM_CLASSES = 6
IMAGE_SIZE = 128

device = "cpu"

model = models.mobilenet_v2(pretrained=False)
model.classifier[1] = nn.Linear(model.last_channel, NUM_CLASSES)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

dummy_input = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE)

torch.onnx.export(
    model,
    dummy_input,
    ONNX_PATH,
    input_names=["input"],
    output_names=["output"],
    opset_version=11
)

print("ONNX export done.")
