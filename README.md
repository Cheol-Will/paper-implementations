# Paper-Implementation

Train a model 
``` bash
    python train.py --model ResNet
    python train.py --model PreActResNet
    python train.py --model DenseNet
    python train.py --model StDepth
    python train.py --model ConvMixer
    python train.py --model MLPMixer
```

``` bash
    python train.py --model ResNet --checkpoint_start_from ConvMixer_1.pth 
```

- Training on CIFAR with batch size 64
- AdamW optimizer and learning schduler with base_lr = 0.0001 to max_lr = 0.001
- augmentation: Horizontal Flip, RandAug, Mixup, CutMix, random erasing

| Model             | Accuracy | Params | Training Time |
|-------------------|----------|--------|---------------|
| ResNet            | 86.66%   | 1.7M   |
| PreActResNet      | 86.46%   | 1.7M   |
| StochasticDepth   | 81.90%   | 1.4M   |
| FractalNet        | -        | 38M    |
| DenseNet          | 85.00%   | 1.4M   |
| ViT               | 58.78%   | 1.5M   |
| ConvNeXt          | -        | -      |
| MLP-Mixer         | 67.38    | 1.6M   |
| ConvMixer         | 91.89%   | 1M     |