# MNIST Classification Models

## Model Architecture Evolution

This repository contains the evolution of CNN models for MNIST classification, with the goal of achieving 99.4% accuracy using less than 8K parameters.

## Setup Instructions

### Prerequisites
- Python 3.8 or higher
- PyTorch 2.0 or higher
- CUDA-capable GPU (optional, for faster training)

### Installation
1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd Assignment-7
   ```

2. Install required packages:
   ```bash
   pip install torch torchvision tqdm matplotlib numpy
   ```

### Running the Model
1. Train the model:
   ```bash
   python Model_4.py
   ```

2. View training progress:
   - Loss and accuracy metrics will be displayed during training
   - Training graphs will be saved in the output directory

### File Structure
```
Assignment-7/
├── Model_4.py        # Latest model implementation
├── utils.py          # Utility functions
└── README.md         # Documentation
```

## Latest Model Results

### Target
- Achieve 99.4% test accuracy
- Keep parameters under 8K
- Complete training in 15 epochs

### Results
- Parameters: 7,918
- Best Test Accuracy: 99.37% (Epoch 13)
- Final Test Accuracy: 99.33% (Epoch 15)
- Training Accuracy: 98.84%

### Training Progress
1. Initial Phase (Epochs 1-3):
   - Epoch 1: 96.66%
   - Epoch 3: 98.66%

2. Mid Phase (Epochs 4-8):
   - Epoch 5: 98.89%
   - Epoch 8: 98.79%

3. Final Phase (Epochs 9-15):
   - Epoch 9: 99.24%
   - Epoch 11: 99.30%
   - Epoch 13: 99.37% (Best)
   - Epoch 15: 99.33%

### Architecture Design
- Input Block: Optimized for initial feature extraction
- Convolution Blocks: Balanced depth and width
- Transition Blocks: Efficient dimensionality reduction
- Output Block: GAP followed by final convolution

### Receptive Field Calculation

Layer-wise receptive field analysis:

```
Layer                   Kernel    Stride    Padding    Input Size    Output Size    RF_in    RF_out
Input                     -         -         -         28x28         28x28          1         1
Conv1 (3x3)              3         1         1         28x28         28x28          1         3
Conv2 (3x3)              3         1         1         28x28         28x28          3         5
Conv3 (3x3)              3         1         1         28x28         28x28          5         7
MaxPool                  2         2         0         28x28         14x14          7         8
Conv4 (1x1)              1         1         0         14x14         14x14          8         8
Conv5 (3x3)              3         1         1         14x14         14x14          8         12
Conv6 (3x3)              3         1         1         14x14         14x14          12        16
GAP                      14        1         0         14x14         1x1            16        28
```

Key Observations:
1. Initial layers (Conv1-Conv3) gradually increase RF from 1 to 7
2. MaxPool layer maintains information while reducing spatial dimensions
3. 1x1 convolution preserves RF while adjusting channels
4. Final layers (Conv5-Conv6) expand RF to capture larger patterns
5. GAP layer ensures global context with 28x28 receptive field

### Training Configuration
- Batch Size: 128
- Optimizer: SGD with momentum
- Learning Rate: Optimized schedule
- Regularization: Dropout and weight decay
- Data Augmentation: Multiple techniques applied

### Analysis

#### Strengths
1. Parameter Efficiency:
   - Successfully maintained under 8K limit (7,918)
   - Good accuracy-to-parameter ratio

2. Training Stability:
   - Consistent improvement across epochs
   - Small gap between train (98.84%) and test (99.33%) accuracy
   - Multiple epochs achieving >99.2%

3. Performance Metrics:
   - Very close to target accuracy (99.37% vs 99.4%)
   - Stable performance in final epochs
   - Quick convergence to high accuracy

#### Areas for Optimization
1. Architecture:
   - Fine-tune channel distribution
   - Optimize transition layers

2. Training:
   - Further learning rate schedule optimization
   - Experiment with regularization parameters

### Conclusion
The model demonstrates excellent parameter efficiency while achieving near-target accuracy. The stable training progression and consistent high performance indicate a well-balanced architecture and training strategy.
