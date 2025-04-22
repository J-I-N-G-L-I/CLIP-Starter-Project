# img.show()
import torchvision.transforms
from torch.utils.data import DataLoader

dataset_transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()
])

test_dataset = torchvision.datasets.CIFAR10("./data", train=False, download=True, transform=torchvision.transforms.ToTensor())

test_loader = DataLoader(test_dataset, batch_size=4, shuffle=True, drop_last=False)

img, target = test_dataset[0]
print(img.shape)
print(target)
print(type(test_loader))
for data in test_loader:
    imgs, targets = data
    print(imgs.shape)
    print(targets)


import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self) -> None:
        super(Model, ).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5)
        self.conv2 = nn.Conv2d(20, 20, 5)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        return F.relu(self.conv2(x))



model = Model()
