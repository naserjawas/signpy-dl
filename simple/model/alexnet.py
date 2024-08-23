from torch.nn import Module
from torch.nn import Sequential
from torch.nn import Conv2d
from torch.nn import ReLU
from torch.nn import MaxPool2d
from torch.nn import Dropout
from torch.nn import Linear

class AlexNet(Module):
    def __init__(self, classes):
        super(AlexNet, self).__init__()
        self.feature = Sequential(
            Conv2d(in_channels=1, out_channels=32, kernel_size=5, stride=1, padding=1),
            ReLU(inplace=True),
            Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=2, stride=2),
            Conv2d(in_channels=64, out_channels=96, kernel_size=3, padding=1),
            ReLU(inplace=True),
            Conv2d(in_channels=96, out_channels=64, kernel_size=3, padding=1),
            ReLU(inplace=True),
            Conv2d(in_channels=64, out_channels=32, kernel_size=3, padding=1),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=2, stride=1),
        )
        self.classifier = Sequential(
            Dropout(),
            Linear(in_features=32*12*12, out_features=2048),
            ReLU(inplace=True),
            Dropout(),
            Linear(in_features=2048, out_features=1024),
            ReLU(inplace=True),
            Linear(in_features=1024, out_features=classes),
        )

    def forward(self, x):
        x = self.feature(x)
        x = x.view(-1, 32*12*12)
        output = self.classifier(x)

        return output
