import torch
import torch.nn as nn

class WinProbabilityModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(5, 16)
        self.layer2 = nn.Linear(16, 16)
        self.layer3 = nn.Linear(16,1)
        

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        x = torch.sigmoid(self.layer3(x))
        return x
