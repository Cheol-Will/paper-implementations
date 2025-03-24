# Paper-Implementation

## Train a model 
To train a model, use the following commands:
``` bash
    python train.py --model {MODEL_NAME}
```

## Training from a Checkpoint
To resume training from a specific checkpoint, use:
``` bash
    python train.py --model ResNet --checkpoint_start_from {CHECKPOINT_PATH} 
```

## Training Setup
- Training on CIFAR with batch size 64
- Adam optimizer with Cosine Annealing Learning Scheduler 
- augmentation: Horizontal Flip, RandAug, Mixup, CutMix, random erasing


## Training Result

| Model             | Accuracy | Params | GFLOPs | Training Time(hours) | 
|-------------------|----------|--------|--------|----------------------|
| ResNet            | 93.13%   | 1.7M   | 0.26   | 2.4                  |
| PreActResNet      | 92.94%   | 1.7M   | 0.26   | 2.1                  |
| StochasticDepth   | 90.65%   | 1.4M   | 0.07   | 1.9                  |
| FractalNet        | 78.20%   | 38M    |        | 2.3                  |
| DenseNet          | 93.36%   | 1.4M   | 0.33   | 1.7                  |
| ViT               | 81.46%   | 1.5M   | 0.16   | 1.3                  |
| MLP-Mixer         | 79.94%   | 1.6M   | 0.30   | 1.3                  |
| ConvMixer         | 95.51%   | 1M     | 1.38   | 5.7                  |