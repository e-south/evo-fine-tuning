import pytest
import torch
from fine_tuning.trainer import train
from fine_tuning.model import DownstreamModel

def test_training_loop():
    input_dim = 256
    output_dim = 2
    model = DownstreamModel(input_dim, output_dim).to("cpu")

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Mock dataset
    inputs = torch.rand(10, input_dim)
    labels = torch.randint(0, output_dim, (10,))
    dataloader = [(inputs, labels)]

    initial_loss = train(model, dataloader, criterion, optimizer, "cpu")
    final_loss = train(model, dataloader, criterion, optimizer, "cpu")

    assert final_loss < initial_loss, "Loss should decrease over epochs"
