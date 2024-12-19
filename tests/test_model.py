import pytest
import torch
from fine_tuning.model import DownstreamModel

def test_downstream_model():
    input_dim = 256
    output_dim = 2
    model = DownstreamModel(input_dim, output_dim)

    inputs = torch.rand(4, input_dim)  # Batch of 4
    outputs = model(inputs)

    assert outputs.shape == (4, output_dim), f"Expected output shape (4, {output_dim}), got {outputs.shape}"
