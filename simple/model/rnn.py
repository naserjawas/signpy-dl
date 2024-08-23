from torch.nn import Module
from torch.nn import LSTM
from torch.nn import BatchNorm1d
from torch.nn import Dropout
from torch.nn import Linear
from torch.nn import ReLU
from torch.nn import LogSoftmax

class RNN(Module):
    def __init__(self, classes):
        super(RNN, self).__init__()

        self.rnn = LSTM(input_size=28, hidden_size=64, batch_first=True)

        self.batchnorm = BatchNorm1d(num_features=64)
        self.dropput1 = Dropout(p=0.25)
        self.fc1 = Linear(in_features=64, out_features=32)
        self.relu1 = ReLU()

        self.dropput2 = Dropout(p=0.5)
        self.fc2 = Linear(in_features=32, out_features=classes)
        self.LogSoftmax = LogSoftmax(dim=1)

    def forward(self, x):
        x = x.reshape(-1, 28, 28)
        x, hidden = self.rnn(x)

        x = x[:, -1, :]
        x = self.batchnorm(x)
        x = self.dropput1(x)
        x = self.fc1(x)
        x = self.relu1(x)

        x = self.dropput2(x)
        x = self.fc2(x)
        output = self.LogSoftmax(x)

        return output
