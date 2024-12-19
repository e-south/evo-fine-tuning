# Entrypoint script for fine-tuning
# qrsh -l h_rt=3:00:00 -pe omp 8 -l gpus=1 -l gpu_c=6.0

import torch
from evo.fine_tuning.data import SequenceDataset
from evo.fine_tuning.model import DownstreamModel
from evo.fine_tuning.trainer import train, evaluate

def main():
    # Configuration
    device = 'cuda:0'
    sequences = ["ACGT", "GCTA", "TACG"]  # Example
    labels = [0, 1, 0]
    tokenizer = ...  # Initialize tokenizer from Evo

    # Dataset and DataLoader
    dataset = SequenceDataset(sequences, labels, tokenizer)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=True)

    # Model
    input_dim = 256  # Example Evo embedding dimension
    output_dim = 2
    model = DownstreamModel(input_dim, output_dim).to(device)

    # Training setup
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Train and evaluate
    for epoch in range(10):
        train(model, dataloader, criterion, optimizer, device)
        loss = evaluate(model, dataloader, criterion, device)
        print(f"Epoch {epoch + 1}, Loss: {loss}")

if __name__ == "__main__":
    main()
