import torch
from torchvision.io import read_image
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights

image = read_image('tench.jpg')
weights = MobileNet_V2_Weights.IMAGENET1K_V2
model = mobilenet_v2(weights=weights)
preprocess = weights.transforms()
batch = preprocess(image).unsqueeze(0)
pred = model(batch).squeeze(0).softmax(0)
index = pred.argmax().item()
proba = pred[index].item()
label = weights.meta['categories'][index]

torch.onnx.export(
    model,
    batch,
    'mobile.onnx',
    verbose=False,
    input_names=['input'],
    output_names=['output'],
    export_params=True,
    opset_version=13
)

