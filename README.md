# Paper-Implementation

# Train a model 
To train a model, use the following commands:
``` bash
    python train.py --model {MODEL_NAME}
```

# Training from a Checkpoint
To resume training from a specific checkpoint, use:
``` bash
    python train.py --model ResNet --checkpoint_start_from {CHECKPOINT_PATH} 
```

# Training Setup
- Training on CIFAR with batch size 64
- Adam optimizer with Cosine Annealing Learning Scheduler 
- augmentation: Horizontal Flip, RandAug, Mixup, CutMix, random erasing

| Model             | Accuracy | Params | GFLOPs | Training Time(hours) | 
|-------------------|----------|--------|--------|----------------------|
| ResNet            | 86.66%   | 1.7M   | 0.26   | 2.4                  |
| PreActResNet      | 86.46%   | 1.7M   | 0.26   | 2.1                  |
| StochasticDepth   | 81.90%   | 1.4M   | 0.07   | 1.9                  |
| FractalNet        | 75.68%   | 38M    |        | 2.3                  |
| DenseNet          | 85.00%   | 1.4M   | 0.33   | 1.7                  |
| ViT               | 58.78%   | 1.5M   | 0.16   | 1.3                  |
| MLP-Mixer         | 67.38    | 1.6M   | 0.30   | 1.3                  |
| ConvMixer         | 91.89%   | 1M     | 1.38   | 5.7                  |