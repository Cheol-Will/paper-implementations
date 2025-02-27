# Paper-Implementation

``` 
    python train.py --model ResNet

```

- Training on CIFAR with batch size 64 and epoch 300
- Learning rate: start from 0.1, and is divided by 10 at 50% and 75% of the total number of training epochs
- weight decay of 10−4 and a Nesterov momentum of 0.9 
- augmentation: mirroring/shifting