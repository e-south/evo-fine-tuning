# Defines models (e.g., Evo wrappers, CNNs)

import torch.nn as nn

class DownstreamModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DownstreamModel, self).__init__()
        self.conv1 = nn.Conv1d(input_dim, 64, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.fc = nn.Linear(64, output_dim)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = x.mean(dim=-1)  # Global average pooling
        x = self.fc(x)
        return x




