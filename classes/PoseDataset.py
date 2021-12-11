import numpy as np
import torch
from torch.utils.data import Dataset
import time

class PoseDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, x,y):
        """
        Args:
            root_dir (string): Directory with all the images.
        """
        self.x  = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        X = torch.FloatTensor(self.x.iloc[idx])
        Y = torch.from_numpy(np.array(self.y.iloc[idx])).type(torch.LongTensor)

        if torch.cuda.is_available():
            X = X.cuda()
            Y = Y.cuda()

        sample = (X,Y)

        return sample