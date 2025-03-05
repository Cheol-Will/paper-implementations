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

| Model             | Accuracy | Avg Loss  | Params |
|-------------------|----------|-----------|--------|
| ResNet            | 84.4%    | 1.005641  | 1.7M   |
| PreActResNet      | 85.7%    | 0.768755  | 1.7M   |
| DenseNet          | 89.9%    | 0.540436  | -      |
| StochasticDepth   | -        | -         | -      |
| FractalNet        | -        | -         | -      |
| MLP-Mixer         | -        | -         | -      |
| ConvMixer         | -        | -         | 1M     |
