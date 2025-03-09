import torch
import torch.nn as nn
import torch.nn.functional as F

class VGGBlock(nn.Module):

    def __init__(self, in_channels, out_channels, stride):
        super(VGGBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        return out

class VGG_34(nn.Module):

    def __init__(self, channel_list, num_blocks_list, num_classes):
        super(VGG_34, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(3, 64, 7, stride=2)
        self.bn1 = nn.BatchNorm2d(64)
        self.vggblock1 = self.make_layer(VGGBlock, channel_list[0], num_blocks_list[0], 1)
        self.vggblock2 = self.make_layer(VGGBlock, channel_list[1], num_blocks_list[1], 2)
        self.vggblock3 = self.make_layer(VGGBlock, channel_list[2], num_blocks_list[2], 2)
        self.vggblock4 = self.make_layer(VGGBlock, channel_list[3], num_blocks_list[3], 2)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(num_blocks_list[3] * 1 * 1, num_classes),
        )

    def make_layer(self, block, out_channels, num_blocks, first_stride):
        stride_list = [first_stride] + [1] * num_blocks
        layers = []
        for stride in stride_list:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels

        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x))) 
        x = F.avg_pool2d(x, 2)
        x = self.vggblock1(x)
        x = self.vggblock2(x)
        x = self.vggblock3(x)
        x = self.vggblock4(x)
        x = self.classifier(x)
        return x