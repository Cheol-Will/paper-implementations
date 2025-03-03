import torch
import torch.nn as nn
import torch.nn.functional as F

class PreActResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(PreActResBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride = stride, padding = 1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride = 1, padding = 1)
        self.shortcut = nn.Sequential()
        
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.BatchNorm2d(in_channels),
                nn.ReLU(),
                nn.Conv2d(in_channels, out_channels, 1, stride = 2),
            )

    def forward(self, x):
        out = self.bn1(x)
        out = self.conv1(F.relu(out))
        out = self.bn2(out)
        out = self.conv2(F.relu(out))
        out += self.shortcut(x)
        return out
    

class PreActBottleNeck(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(PreActBottleNeck, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 1, stride = stride)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride = stride)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels * 4, 1, stride = stride)
        self.shortcut = nn.Sequential()
        
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.BatchNorm2d(out_channels*4),
                nn.ReLU(),
                nn.Conv2d(in_channels, out_channels*4, 1, stride = stride),
            )

    def forward(self, x):
        out = self.bn1(x)
        out = self.conv1(F.relu(out))
        out = self.bn2(out)
        out = self.conv2(F.relu(out))
        out = self.bn3(out)
        out = self.conv3(F.relu(out))
        out += self.shortcut(x)
        return out
    
class PreActResNet(nn.Module):
    def __init__(self, block, num_blocks_list, increase_rate = 1, dataset = "cifar-10"):
        super(PreActResNet, self).__init__()
        self.increase_rate = increase_rate

        if dataset == "cifar-10":
            self.conv1 = nn.Conv2d(3, 16, 3, stride = 1, padding = 1)
            self.in_channels = 16
            self.conv = nn.Sequential(
                self.make_layer(block, 16, num_blocks_list[0], 1),
                self.make_layer(block, 32, num_blocks_list[1], 2),
                self.make_layer(block, 64, num_blocks_list[2], 2),
            )
            self.in_channels = 64
            self.num_class = 10

        elif data == "imagenet":
            self.conv1 = nn.Conv2d(3, 64, 7, stride = 2)
            self.in_channels = 64
            # need to add BatchNorm
            self.conv = nn.Sequential(
                nn.AvgPool2d(2),
                self.make_layer(block, 64, num_blocks_list[0], 1),
                self.make_layer(block, 128, num_blocks_list[2], 2),
                self.make_layer(block, 256, num_blocks_list[3], 2),
                self.make_layer(block, 512, num_blocks_list[4], 2),
            )
            self.in_channels = 512
            self.num_class = 1000
        
        # Last BN and ReLU is added bleow since layers in block are re-arranged
        self.bn = nn.BatchNorm2d(self.in_channels * increase_rate)
        self.relu = nn.ReLU()
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(self.in_channels * 1 * 1 * increase_rate, self.num_class),
            nn.LogSoftmax(dim = 1) 
        )

    def make_layer(self, block, out_channels, num_blocks, first_stride):
        stride_list = [first_stride] + [1] * (num_blocks - 1)
        layers = []
        
        # create layers with given block (eg. PreActResBlock or PreActBottleNeck)
        for stride in stride_list:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * self.increase_rate

        for layer in layers:
            if type(layer) == nn.Conv2d:
                nn.init.kaiming_normal_(layer.weifght, a = 0, mode = "fan_in", nonlinearity='leaky_relu', generator=None)

        return nn.Sequential(*layers)


    def forward(self, x):
        x = self.conv1(x)
        x = self.conv(x)
        x = self.relu(self.bn(x))
        x = self.classifier(x)
        return x