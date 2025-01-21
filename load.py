import torch
from preprocess.preprocess import ImageDataset, displayImage
from unet import UNet
import random

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def testModel():
    model = UNet()
    model.load_state_dict(torch.load("unet_model.pth", weights_only=True))
    model.eval()
    model = model.to(device)

    dataset = ImageDataset()
    for _ in range(10):
        idx = random.randint(0, len(dataset))
        x, y = dataset[idx]['X'], dataset[idx]['Y']
        x = x[None, :, :, :]
        y = y[None, :, :, :]

        x = x.to(device=device, dtype=torch.float32)

        mask, _ = model(x)
        displayImage(x.cpu().detach())
        displayImage(torch.sigmoid(mask).cpu().detach())

testModel()

