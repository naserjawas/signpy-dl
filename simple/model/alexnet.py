from torch.nn import Module
from torch.nn import Sequential
from torch.nn import Conv2d
from torch.nn import ReLU
from torch.nn import MaxPool2d
from torch.nn import AdaptiveAvgPool2d
from torch.nn import Dropout
from torch.nn import Linear
from torch import flatten


class AlexNet(Module):
    def __init__(self, classes):
        super(AlexNet, self).__init__()

        self.features = Sequential(
            Conv2d(in_channels=3, out_channels=64, kernel_size=11, stride=4, padding=2),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=3, stride=2),
            Conv2d(in_channels=64, out_channels=192, kernel_size=5, padding=2),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=3, stride=2),
            Conv2d(in_channels=192, out_channels=384, kernel_size=3, padding=1),
            ReLU(inplace=True),
            Conv2d(in_channels=384, out_channels=256, kernel_size=3, padding=1),
            ReLU(inplace=True),
            Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = AdaptiveAvgPool2d((6, 6))
        self.classifier = Sequential(
            Dropout(p=0.5),
            Linear(in_features=256 * 6 * 6, out_features=4096),
            ReLU(inplace=True),
            Dropout(p=0.5),
            Linear(in_features=4096, out_features=4096),
            ReLU(inplace=True),
            Linear(in_features=4096, out_features=classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = flatten(x, 1)
        output = self.classifier(x)

        return output
