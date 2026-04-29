# CIFAR-10 and CIFAR-100 Classification - Correct Implementation

This is a clean, well-structured implementation of CIFAR-10 image classification using PyTorch. It supports both Multi-Layer Perceptron (MLP) and Convolutional Neural Network (CNN) architectures. The project is also extended in Task 2 to support the more challenging CIFAR-100 dataset.

## Project Structure

```
.
├── config.py           # Configuration settings
├── models.py           # Model architectures (MLP and CNN)
├── data_loader.py      # Data loading and preprocessing
├── utils.py            # Training and evaluation utilities
├── train.py            # Main training script
├── requirements.txt    # Python dependencies
└── readme.md           # This file
└── changelog.md        # Shows bug fixes for Task1
└── PerformanceReport.md           # Performance Report for Task1 for MLP and CNN (How each model performed after bug fixes)
└── ablation_study.md           # Shows log for experimenation
```

## Dataset Support

This codebase supports both CIFAR-10 and CIFAR-100 datasets.

- **CIFAR-10**: 10 classes (used in Task 1 for code stabilization)
- **CIFAR-100**: 100 classes (used in Task 2 for adaptation and optimization)

The dataset is selected internally via the data loader. For CIFAR-100 experiments, the number of output classes is set to 100 and the CIFAR-100 dataset is used.


## Setup

1. **Install dependencies:**

See *requirements.txt*.

2. **Verify installation:**
```bash
python models.py
python data_loader.py
```

## Usage

### Quick Start

Run training with default settings (CNN model):
```bash
python train.py
```

### Customizing Configuration

Edit `config.py` to change settings.

### Model Architectures

**MLP (Multi-Layer Perceptron):**
- Input: Flattened 32×32×3 images 
- Hidden layers: [512, 256] (configurable)
- BatchNorm + ReLU + Dropout after each hidden layer
- Output: 10 classes

**CNN (Convolutional Neural Network):**
- 2 convolutional blocks (32→64)
- Each block: 2 Conv layers + BatchNorm + ReLU + Dropout + MaxPool 
- For Task2 Added one additional Convolutional Block
- Fully connected layers: 512 → number of classes (10 for CIFAR-10, 100 for CIFAR-100)

### Expected Performance

These are the results for Task1 (MLP and CNN for CIFAR-10) and Task2 (CNN only)

| Model | Validation Acc | Test Acc | Training Time* |
|-------|---------------|----------|----------------|
| MLP CIFAR-10    | 50.22%     | 51.28%   | ~2 min     |
| CNN CIFAR-10    | 77.10%     | 76.44%   | ~2 min    |
| CNN CIFAR-100   | 61.92%     | 63.67%   | ~15 min    |

*On GPU (NVIDIA RTX 3080)

## Output Files

After training, the following files are created in `./checkpoints/`:

- `best_model_cnn.pth` - Best model checkpoint
- `training_history_cnn.png` - Loss and accuracy curves
- `evaluation.py` - Standalone script to load the final checkpoint and evaluate it on the CIFAR-100 test set

## Final Model Checkpoint and Evaluation (Task 2)

The best-performing CNN model trained on CIFAR-100 is saved as a checkpoint:

- `./checkpoints/best_model_cnn.pth`

This checkpoint corresponds to the model state that achieved the highest validation accuracy during Task 2 experiments.

### Evaluation Procedure

A standalone evaluation script is provided to reproduce the reported test performance:
python code/evaluation.py


## Citation

Dataset: [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html)
- Learning Multiple Layers of Features from Tiny Images, Alex Krizhevsky, 2009.

## License

This code is for educational purposes.
