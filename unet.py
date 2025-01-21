import torch
from torch import optim
import torch.nn as nn
from torch.nn import functional as F
from preprocess.preprocess import ImageDataset, displayImage
from torch.utils.data import DataLoader, random_split

MAX_ITERS = 5000
eval_interval = 500
learning_rate = 1e-7
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200

def dice_loss_binary(prediction, targets, smooth=1e-6):
    probs = torch.sigmoid(prediction)
    probs_flat = probs.view(-1)
    targets_flat = targets.view(-1)
    
    intersection = (probs_flat * targets_flat).sum()
    dice_coeff = (2.0 * intersection + smooth) / (probs_flat.sum() + targets_flat.sum() + smooth)

    return 1 - dice_coeff

class EncodingLayer(nn.Module):

    def __init__(self, inChannels, features=None, first=False, last=False):

        super().__init__()

        if features is None:
            features = 2 * inChannels

        self.first = first
        self.nn = nn.Sequential(
            nn.Conv2d(inChannels, features, (3, 3), padding=1),
            nn.ReLU(),
            nn.Conv2d(features, features if not last else inChannels, (3, 3), padding=1),
            nn.ReLU(),
        )
        
        if not self.first:
            self.maxPool = nn.MaxPool2d((2, 2), 2)

    def forward(self, x):

        if not self.first:
            x = self.maxPool(x)

        x = self.nn(x)
        return x
    

class Encoder(nn.Module):

    def __init__(self):

        super().__init__()

        self.nn = nn.Sequential(
            EncodingLayer(3, 64, first=True),
            EncodingLayer(64),
            EncodingLayer(128),
            EncodingLayer(256),
            EncodingLayer(512, last=True),
        )
    
    def forward(self, x):
        outputs = []
        for layer in self.nn:
            x = layer(x)
            outputs.append(x)
            print(f"Encoder Layer output min: {x.min().item()}, max: {x.max().item()}, mean: {x.mean().item()}")
        return outputs[::-1] # 0th index is the output for the last layer

class DecodingLayer(nn.Module):

    def __init__(self, inChannels, features = None):

        super().__init__()

        self.features = features
        if self.features is None:
            self.features = inChannels // 2

        self.nn = nn.Sequential(
            nn.Conv2d(2 * inChannels, inChannels, (3, 3), padding=1),
            nn.ReLU(),
            nn.Conv2d(inChannels, self.features, (3, 3), padding=1),
            nn.ReLU()
        )

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners = True)

    def forward(self, x1, x2):
        x1 = self.upsample(x1)

        if x1.shape != x2.shape: # pad to make sure both are same size per channel (h, w), this happens when input image not power of 2 since we half the dimensions with maxpool
            x_, y_= x2.shape[3] - x1.shape[3], x2.shape[2] - x1.shape[2]
            x1 = F.pad(x1, (x_ // 2, x_ - x_ // 2, y_ // 2, y_ - y_ // 2))

        x = torch.cat((x1, x2), dim=1) # add across channel dim, this is why its 2*inChannels for first convolution, the second input is the corresponding encoder output
        x = self.nn(x)
        return x
        
class Decoder(nn.Module):

    def __init__(self):

        super().__init__()

        self.nn = nn.Sequential(
            DecodingLayer(512),
            DecodingLayer(256),
            DecodingLayer(128),
            DecodingLayer(64, 64)
        )
    def forward(self, outputs):

        x = outputs[0]

        for i in range(len(self.nn)):
            layer = self.nn[i]
            x = layer(x, outputs[i + 1])
            print(f"Decoder Layer output min: {x.min().item()}, max: {x.max().item()}, mean: {x.mean().item()}")

        return x
    
class UNet(nn.Module):

    def __init__(self):

        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.final = nn.Conv2d(64, 1, (3, 3), padding=1)
        self.criterion = nn.BCEWithLogitsLoss()

    def forward(self, input, target = None):
        x = self.encoder(input)
        x = self.decoder(x)
        prediction = self.final(x)
        
        loss = None
        if target is not None:
            loss = self.criterion(prediction, target)

        return prediction, loss

model = UNet()
model = model.to(device)
optimizer = optim.RMSprop(model.parameters(),
                            lr=learning_rate, weight_decay=1e-8, momentum=0.99, foreach=True)

dataset = ImageDataset()
num_epochs = 10
def train():
    model.train()
    numEval = len(dataset) // 10
    numTrain = len(dataset) - numEval

    trainSet, valSet = random_split(dataset, [numTrain, numEval], generator=torch.Generator().manual_seed(0))

    trainLoader = DataLoader(trainSet, shuffle=True, batch_size=1)
    valLoader = DataLoader(valSet, shuffle=False, drop_last=True, batch_size=1)
    for epoch in range(num_epochs):

        for iter, batch in enumerate(trainLoader):
            images, masks = batch['X'], batch['Y']
            images = images.to(device=device, dtype=torch.float32)
            masks = masks.to(device=device, dtype=torch.float32)

            # if iter != 0 and iter % eval_interval == 0: # test the model every eval intervals
            #     model.eval()
            #     losses = torch.zeros(len(valLoader))
            #     for i, valBatch in enumerate(valLoader):
            #         valX, valY= valBatch['X'], valBatch['Y']

            #         X = valX.to(device=device, dtype=torch.float32)
            #         Y = valY.to(device=device, dtype=torch.float32)

            #         predictions, loss = model(X, Y)
            #         losses[i] = loss.item()

            #     meanLoss = losses.mean() # do eval iters iterations of testing, and take average for better loss calculation
            #     print(f"step {iter}: val loss {meanLoss:.4f}")

            #     model.train()

            predictions, loss = model(images, masks)
            probs = torch.sigmoid(predictions)

            print(f"Sigmoid probs min: {probs.min().item()}, max: {probs.max().item()}, mean: {probs.mean().item()}")
            print(f"Logits min: {predictions.min().item()}, max: {predictions.max().item()}, mean: {predictions.mean().item()}")

            print(f"Loss before dice: {loss.item()}")
            loss += dice_loss_binary(predictions, masks)

            print(f"Iter {iter}, training loss = {loss.item()}")

            optimizer.zero_grad(set_to_none=True)
            loss.backward()

            optimizer.step()

        displayImage(torch.sigmoid(predictions).cpu().detach())
        displayImage(masks.cpu().detach())


if __name__ == "__main__":
    train()

    save_path = "unet_model.pth"

    torch.save(model.state_dict(), save_path)
    print(f"Model weights saved to {save_path}")

    model.eval()

    sample = next(iter(dataset))

    x = sample['X']
    y = sample['Y']
    y_pred = model(x)
    y_pred = torch.sigmoid(y_pred).cpu().detach()

    displayImage(x)
    displayImage(y)
    displayImage(y_pred)