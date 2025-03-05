import torch
import torch.nn as nn
import torch.nn.functional as F

class DenseBlock(nn.Module):

    def __init__(self, in_channels, k, num_blocks):
        """
            k: growth rate
            num_blocks: the number of layers in block
        """
        super(DenseBlock, self).__init__()
        self.in_channels = in_channels
        self.growth_rate = k
        self.num_blocks = num_blocks
        self.block = self.make_block()

    def make_block(self):
        layers = []
        for _ in range(self.num_blocks):
        
            # BatchNorm - ReLU - 3x3Conv 
            layers.append(nn.Sequential(
                nn.BatchNorm2d(self.in_channels),
                nn.ReLU(),
                nn.Conv2d(self.in_channels, self.growth_rate, 3, padding = 1),
            ))

            # next layer gets k more input feature maps
            self.in_channels += self.growth_rate

        # weight initialization
        for layer in layers:
            if type(layer) == nn.Conv2d:
                nn.init.kaiming_normal_(layer.weight, a=0, mode='fan_in', nonlinearity='leaky_relu', generator=None)

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
        self.num_blocks = num_blocks
        self.block = self.make_block()
        pass

    def make_block(self):
        layers = []
        for _ in range(self.num_blocks):

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

        # weight initialization
        for layer in layers:
            if type(layer) == nn.Conv2d:
                nn.init.kaiming_normal_(layer.weight, a=0, mode='fan_in', nonlinearity='leaky_relu', generator=None)
        
        return nn.Sequential(*layers)

    def forward(self, x):
        for layer in self.block:
            x = torch.concat([x, layer(x)], dim = 1)

        return x

class DenseNet(nn.Module):

    def __init__(self, block, num_blocks_list, growth_rate = 32, in_channels = 16, compression_rate = 1):
        super(DenseNet, self).__init__()
        self.growth_rate = growth_rate
        self.compression_rate = compression_rate
        self.in_channels = in_channels

        self.conv1 = nn.Conv2d(3, self.in_channels, 3, stride = 1, padding = 1)
        # self.pool1 = nn.MaxPool2d(3, stride=2)
        self.dense = self.make_layer(block, num_blocks_list)
        self.clf = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(self.in_channels, 10), 
            nn.LogSoftmax(dim = 1) 
        )
    def make_layer(self, block, num_blocks_list):
        layers = []
        for idx, num_blocks in enumerate(num_blocks_list):
            b = block(self.in_channels, self.growth_rate, num_blocks)
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
        # x = self.pool1(x)
        x = self.dense(x)
        x = self.clf(x)
        return x