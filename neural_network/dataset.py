import torch
from torch.utils.data import Dataset

class RegressionDataset(Dataset):
    def __init__(self, features, targets):
        self.features = features
        self.targets = targets

    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, index):
        fingerprint = torch.tensor(self.features[index], dtype=torch.float)
        logd74 = torch.tensor(self.targets[index], dtype=torch.float)

        return fingerprint, logd74
    
    def add(self, feature, target):
        self.features.append(feature)
        self.targets.append(target)
