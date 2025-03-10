import torch
import torch.nn as nn
import torch.nn.functional as F

class PreActResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(PreActResBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.shortcut = nn.Sequential()
        
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.BatchNorm2d(in_channels),
                nn.ReLU(),
                nn.Conv2d(in_channels, out_channels, 1, stride=2),
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
        self.conv1 = nn.Conv2d(in_channels, out_channels, 1, stride=stride)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride=stride)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels * 4, 1, stride=stride)
        self.shortcut = nn.Sequential()
        
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.BatchNorm2d(out_channels*4),
                nn.ReLU(),
                nn.Conv2d(in_channels, out_channels*4, 1, stride=stride),
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
    def __init__(self, block, channel_list, num_blocks_list, increase_rate=1, dataset="cifar-10"):
        super(PreActResNet, self).__init__()
        self.increase_rate = increase_rate

        if dataset == "cifar-10":
            self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
            self.in_channels = 16
            self.conv = nn.Sequential(
                self.make_layer(block, channel_list[0], num_blocks_list[0], 1),
                self.make_layer(block, channel_list[1], num_blocks_list[1], 2),
                self.make_layer(block, channel_list[2], num_blocks_list[2], 2),
            )
            self.in_channels = 64
            self.num_classes = 10

        elif dataset == "imagenet":
            self.conv1 = nn.Conv2d(3, 64, 7, stride=2)
            self.in_channels = 64
            # need to add BatchNorm
            self.conv = nn.Sequential(
                nn.AvgPool2d(2),
                self.make_layer(block, channel_list[0], num_blocks_list[0], 1),
                self.make_layer(block, channel_list[1], num_blocks_list[1], 2),
                self.make_layer(block, channel_list[2], num_blocks_list[2], 2),
                self.make_layer(block, channel_list[3], num_blocks_list[3], 2),
            )
            self.in_channels = 512
            self.num_classes = 1000
        
        # Last BN and ReLU is added bleow since layers in block are re-arranged
        self.bn = nn.BatchNorm2d(self.in_channels * increase_rate)
        self.relu = nn.ReLU()
        self.clf = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(self.in_channels * 1 * 1 * increase_rate, self.num_classes),
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
                nn.init.kaiming_normal_(layer.weight)

        return nn.Sequential(*layers)


    def forward(self, x):
        x = self.conv1(x)
        x = self.conv(x)
        x = self.relu(self.bn(x))
        x = self.clf(x)
        return x