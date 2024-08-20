import torch.nn as nn

class MLP_Lipo(nn.Module):
    def __init__(self):
        super(MLP_Lipo, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(2048, 512),  nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(512, 256),   nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, 16),    nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(16, 1),
        )

    def forward(self, x):
        x = self.model(x)
        x = x.squeeze(1)

        return x

class MLP_Abalone(nn.Module):
    def __init__(self):
        super(MLP_Abalone, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(8, 6),   nn.LeakyReLU(), nn.Dropout(0.2),
            nn.Linear(6, 4),   nn.LeakyReLU(), nn.Dropout(0.2),
            nn.Linear(4, 2),   nn.LeakyReLU(), nn.Dropout(0.2),
            nn.Linear(2, 1),
        )

    def forward(self, x):
        x = self.model(x)
        x = x.squeeze(1)

        return x

class MLP_Inhibition(nn.Module):
    def __init__(self):
        super(MLP_Inhibition, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(29, 15),   nn.LeakyReLU(), nn.Dropout(0.2),
            nn.Linear(15, 7),   nn.LeakyReLU(), nn.Dropout(0.2),
            nn.Linear(7, 3),   nn.LeakyReLU(), nn.Dropout(0.2),
            nn.Linear(3, 1),
        )

    def forward(self, x):
        x = self.model(x)
        x = x.squeeze(1)

        return x