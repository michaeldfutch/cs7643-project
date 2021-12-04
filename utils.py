# placeholder for utils
# probably ideally we write some functions to download the data and store it locally
# where it won't be committed to the repo

import torch
import torch.nn as nn

## starting with the vanilla CNN from assignment 2
class VanillaCNN(nn.Module):
    def __init__(self):
        super(VanillaCNN, self).__init__()
        #############################################################################
        # TODO: Initialize the Vanilla CNN                                          #
        #       Conv: 7x7 kernel, stride 1 and padding 0                            #
        #       Max Pooling: 2x2 kernel, stride 2                                   #
        #############################################################################
        

        self.model = nn.Sequential(
            nn.Conv2d(in_channels=3,out_channels=32, kernel_size=7, stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(1),
            nn.Linear(5408,out_features=10)
        )
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################

    def forward(self, x):
        outs = None
        #############################################################################
        # TODO: Implement forward pass of the network                               #
        #############################################################################
        x = torch.reshape(x,(x.shape[0], 3, 32, 32))
        outs = self.model(x)
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################

        return outs

