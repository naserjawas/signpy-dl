import torch.nn as nn

class RNN(nn.Module):
    def __init__(self, classes):
        super(RNN, self).__init__()

        self.rnn = nn.LSTM(input_size=28, hidden_size=64, batch_first=True)

        self.batchnorm = nn.BatchNorm1d(num_features=64)
        self.dropput1 = nn.Dropout(p=0.25)
        self.fc1 = nn.Linear(in_features=64, out_features=32)
        self.relu1 = nn.ReLU()

        self.dropput2 = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(in_features=32, out_features=classes)
        self.LogSoftmax = nn.LogSoftmax(dim=1)

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
