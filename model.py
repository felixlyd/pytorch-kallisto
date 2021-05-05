import torch.nn as nn


class Kallisto(nn.Module):
    def __init__(self):
        super(Kallisto, self).__init__()
        self.softmax = nn.Softmax()

    def forward(self,x):
        x = self.softmax(x)
        return x
