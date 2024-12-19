# Helper functions for logging, metrics, etc.

def calculate_accuracy(predictions, labels):
    preds = predictions.argmax(dim=1)
    return (preds == labels).float().mean().item()
