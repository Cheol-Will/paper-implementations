# Paper-Implementation

``` bash
    python train.py --model ResNet
    python train.py --model PreActResNet
    python train.py --model DenseNet
    python train.py --model StDepth
    python train.py --model ConvMixer
    python train.py --model MLPMixer
```

- Training on CIFAR with batch size 64 and epoch 300
- Learning rate: start from 0.1, and is divided by 10 at 50% and 75% of the total number of training epochs
- weight decay of 10−4 and a Nesterov momentum of 0.9 
- augmentation: mirroring/shifting

| Model             | Accuracy | Params |
|-------------------|----------|--------|
| ResNet            | 87.8%    | 1.7M   |
| PreActResNet      | 86.4%    | 1.7M   |
| DenseNet          | 89.9%    | -      |
| StochasticDepth   | -        | -      |
| FractalNet        | -        | -      |
| MLP-Mixer         | -        | -      |
| ConvMixer         | 89.6%    | 1M     |
