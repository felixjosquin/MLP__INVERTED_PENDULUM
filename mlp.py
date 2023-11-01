import torch
import torch.nn as nn


class SimpleMLP(nn.Module):
    def __init__(self, hidden_size):
        super(SimpleMLP, self).__init__()
        self.fc1 = nn.Linear(4, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


model = SimpleMLP(10)
