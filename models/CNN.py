import torch
import torch.nn as nn

# using vanilla CNN from assignment 2 as a starting point
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels=3,out_channels=32, kernel_size=7, stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(1),
            nn.Linear(5408,out_features=20)
        )


    def forward(self, x):
        outs = None
        x = torch.reshape(x,(x.shape[0], 3, 32, 32))
        outs = self.model(x)
        return outs