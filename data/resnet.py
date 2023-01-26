import torch
import torchvision
from torchvision.models import resnet18, ResNet18_Weights

model = torchvision.models.resnet18(weights=ResNet18_Weights.DEFAULT)
model.eval()
sample = torch.rand(1, 3, 224, 224)

model = model.to('cpu')
smaple = sample.to('cpu')
traced_script_module = torch.jit.trace(model, sample)
traced_script_module.save('resnet.pt')
