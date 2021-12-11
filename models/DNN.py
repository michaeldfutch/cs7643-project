import torch
import torch.nn as nn
import torch.nn.functional as F


class DNN(nn.Module):

    def __init__(self,numClasses = 80,features=75,hidden_dim=128):
        super(DNN, self).__init__()

        self.fc1 = nn.Linear(features,hidden_dim)
        self.fc2 = nn.Linear(hidden_dim,numClasses)

    def forward(self, x):
        x = self.fc1(x)
        x = torch.sigmoid(x)
        x = self.fc2(x)

        out = x
        return out
