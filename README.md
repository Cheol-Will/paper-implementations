# Paper-Implementation

``` bash
    python train.py --model ResNet
    python train.py --model PreActResNet
    python train.py --model DenseNet
    python train.py --model StDepth
    python train.py --model ConvMixer
    python train.py --model MLPMixer
```

- Training on CIFAR with batch size 64
- AdamW optimizer and learning schduler with base_lr = 0.0001 to max_lr = 0.001
- augmentation: Horizontal Flip, RandAug, Mixup, CutMix, random erasing

| Model             | Accuracy | Params |
|-------------------|----------|--------|
| ResNet            | 87.8%    | 1.7M   |
| PreActResNet      | 86.4%    | 1.7M   |
| DenseNet          | 89.9%    | -      |
| StochasticDepth   | -        | -      |
| FractalNet        | -        | -      |
| MLP-Mixer         | -        | -      |
| ConvMixer         | 89.6%    | 1M     |