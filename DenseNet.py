import torch
import torch.nn as nn
import torch.functional as F

class DenseBlock(nn.Module):

    def __init__(self, in_channels, k, num):
        """
            k: growth rate
            num: the number of layers in block (l)
        """
        super(DenseBlock, self).__init__()
        self.in_channels = in_channels
        self.growth_rate = k
        self.num = num
        self.block = self.make_block()

    def make_block(self):
        layers = []
        for i in range(self.num):
        
            # BatchNorm - ReLU - 3x3Conv 
            layers.append(nn.Sequential(
                nn.BatchNorm2d(self.in_channels),
                nn.ReLU(),
                nn.Conv2d(self.in_channels, self.growth_rate, 3, padding = 1),
            ))
            # next layer gets k more input feature maps
            self.in_channels += self.growth_rate

        return nn.Sequential(*layers)

    def forward(self, x):
        for layer in self.block:
            x = torch.concat([x, layer(x)], dim = 1)

        return x

class DenseBottleneck(nn.Module):
    def __init__(self, in_channels, k, num):
        super(DenseBottleneck, self).__init__()
        self.in_channels = in_channels
        self.growth_rate = k
        self.num = num
        self.block = self.make_block()
        pass

    def make_block(self):
        layers = []
        for i in range(self.num):

            # BatchNorm - ReLU - 1x1Conv(4k) - BatchNorm - ReLU - 3x3Conv(k) 
            layers.append(nn.Sequential( 
                nn.BatchNorm2d(self.in_channels),
                nn.ReLU(),
                nn.Conv2d(self.in_channels, self.growth_rate*4, 1),
                nn.BatchNorm2d(self.growth_rate*4),
                nn.ReLU(),
                nn.Conv2d(self.growth_rate*4, self.growth_rate, 3, padding = 1),
            ))
            # next layer gets k more input feature maps
            self.in_channels += self.growth_rate

        return nn.Sequential(*layers)

    def forward(self, x):
        for layer in self.block:
            x = torch.concat([x, layer(x)], dim = 1)

        return x

class DenseNet(nn.Module):

    def __init__(self, block, num_blocks_list, growth_rate = 32, in_channels = 64, compression_rate = 1):
        super(DenseNet, self).__init__()
        self.growth_rate = growth_rate
        self.in_channels = in_channels
        self.compression_rate = compression_rate

        self.conv1 = nn.Conv2d(in_channels, 2*growth_rate, 7, s = 2)
        self.pool1 = nn.MaxPool2d(3, stride=2)
        self.dense = self.make_layer(block, num_blocks_list)
        self.clf = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(self.in_channels, 1000), 
            nn.LogSoftmax(dim = 0) 
        )
    def make_layer(self, block, num_blocks_list):
        layers = []
        for idx, num in enumerate(num_blocks_list):
            b = block(self.in_channels, self.growth_rate, num)
            layers.append(b)
            self.in_channels = b.in_channels
            
            if idx == len(num_blocks_list):
                continue

            # transition layer for down sampling and half feature map size
            layers.append(nn.Sequential(
                nn.Conv2d(self.in_channels, self.in_channels * self.compression_rate, 1),
                nn.AvgPool2d(2, stride = 2)
            ))
            # modify in_channels for next blocks
            self.in_channels = b.in_channels * self.compression_rate

        return nn.Sequential(*layers)


    def forward(self, x):
        x = self.conv1(x)
        x = self.poo1(x)
        x = self.dense(x)
        x = self.clf(x)
        return x

