"""
Evaluate a trained CNN checkpoint on CIFAR-100 test set.
"""

import torch
import torch.nn as nn
from config import Config
from models import get_model
from data_loader import get_data_loaders
from utils import evaluate

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load test data only
    _, _, test_loader = get_data_loaders(
        data_dir=Config.DATA_DIR,
        batch_size=Config.BATCH_SIZE,
        val_split=Config.VAL_SPLIT,
        num_workers=Config.NUM_WORKERS,
    )

    # Create model
    model = get_model("cnn", num_classes=Config.NUM_CLASSES)
    model.to(device)

    # Load checkpoint
    checkpoint_path = "checkpoints/best_model_cnn.pth"
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    print("Checkpoint loaded successfully.")

    # Loss
    criterion = nn.CrossEntropyLoss()

    # Evaluate
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)

    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.2f}%")

if __name__ == "__main__":
    main()
