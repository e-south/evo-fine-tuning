import pytest
from fine_tuning.data import SequenceDataset
from torch.utils.data import DataLoader

def test_sequence_dataset():
    sequences = ["ACGT", "GCTA", "TACG"]
    labels = [0, 1, 0]

    dataset = SequenceDataset(sequences, labels, tokenizer=None)  # Replace `None` with a dummy tokenizer
    dataloader = DataLoader(dataset, batch_size=2)

    assert len(dataset) == 3, "Dataset should contain 3 items"

    for batch in dataloader:
        assert "sequences" in batch and "labels" in batch, "Batch should contain sequences and labels"
