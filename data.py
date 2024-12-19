# Handles dataset preparation and loading

import torch
from torch.utils.data import Dataset

class SequenceDataset(Dataset):
    def __init__(self, sequences, labels, tokenizer):
        self.sequences = sequences
        self.labels = labels
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        label = self.labels[idx]
        input_ids = torch.tensor(
            self.tokenizer.tokenize(sequence),
            dtype=torch.int,
        ).unsqueeze(0)  # Add batch dimension
        return input_ids, label
