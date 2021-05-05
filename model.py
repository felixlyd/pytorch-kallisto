import torch
import torch.nn as nn


class Kallisto(nn.Module):
    def __init__(self, input_size):
        super(Kallisto, self).__init__()
        embed_dim = 1
        self.embedding = nn.Embedding(input_size, embed_dim, padding_idx=input_size - 1)
        self.softmax = nn.Softmax(dim=0)

    def forward(self, x):
        # x.size [len(trans),]
        x = self.embedding(x)
        # x.size [len(trans), embed_dim]
        x = self.softmax(x)
        return x


class KallistoA(nn.Module):
    def __init__(self, input_size, active):
        super(KallistoA, self).__init__()
        embed_dim = 1
        self.embedding = nn.Embedding(input_size, embed_dim, padding_idx=input_size - 1)
        if active == "NULL":
            self.func = None
        else:
            func = getattr(nn, active)
            self.func = func()

    def forward(self, x):
        # x.size [len(trans),]
        x = self.embedding(x)
        # x.size [len(trans), embed_dim]
        if self.func is not None:
            x = self.func(x)
        x = torch.div(x, torch.sum(x))
        return x