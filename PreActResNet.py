import torch
import torch.nn as nn
import torch.functional as F

class PreActResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(PreActResBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride = stride)
        self.bn2 = nn.BatchNorm2d()
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride = stride)
        self.shortcut = nn.Sequential()
        
        if in_channels != out_channels:
            self.shortcut = nn.Sequential([
                nn.BatchNorm2d(in_channels),
                nn.ReLU(),
                nn.Conv2d(in_channels, out_channels, 1, stride = 2),
            ])

    def forward(self, x):
        out = self.bn1(x)
        out = self.conv1(F.relu(out))
        out = self.bn2(x)
        out = self.conv2(F.relu(out))
        out += self.shortcut(x)
        return 
    

class PreActBottleNeck(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(PreActResBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 1, stride = stride)
        self.bn2 = nn.BatchNorm2d()
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride = stride)
        self.bn3 = nn.BatchNorm2d()
        self.conv3 = nn.Conv2d(out_channels, out_channels * 4, 1, stride = stride)
        self.shortcut = nn.Sequential()
        
        if in_channels != out_channels:
            self.shortcut = nn.Sequential([
                nn.BatchNorm2d(out_channels*4),
                nn.ReLU(),
                nn.Conv2d(in_channels, out_channels*4, 1, stride = 2),
            ])

    def forward(self, x):
        out = self.bn1(x)
        out = self.conv1(F.relu(out))
        out = self.bn2(x)
        out = self.conv2(F.relu(out))
        out = self.bn3(x)
        out = self.conv3(F.relu(out))
        out += self.shortcut(x)
        return 
    
class PreActResNet(nn.Module):
    def __init__(self, block, num_blocks_list, increase_rate = 1, in_channels = 64):
        super(PreActResNet, self).__init__()
        self.conv1 = nn.Conv2d()
        self.increase_rate = increase_rate
        self.in_channels = in_channels

        self.conv1 = nn.Conv2d(3, in_channels, 7, stride = 2)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.AvgPool2d(2, 2)
        self.conv2_x = self.make_layer(block, 64, num_blocks_list[0], 1)
        self.conv3_x = self.make_layer(block, 128, num_blocks_list[1], 2)
        self.conv4_x = self.make_layer(block, 256, num_blocks_list[2], 2)
        self.conv5_x = self.make_layer(block, 512, num_blocks_list[3], 2)

        # Last BN and ReLU is added bleow since layers in block are re-arranged
        self.bn2 = nn.BatchNorm2d()
        self.relu2 = nn.ReLU()
        self.classifier = nn.Sequential(
            nn.AvgPool2d(kernel_size  = 7),
            nn.Linear(512 * 1 * 1 * increase_rate, 1000),
            nn.LogSoftmax(dim = 0) 
        )

    def make_layer(self, block, out_channels, num_blocks, first_stride):
        layers = []
        stride_list = [first_stride] + [1] * num_blocks
        
        for stride in stride_list:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * self.increase_rate

        for layer in layers:
            if type(layer) == nn.Conv2d:
                nn.init.kaiming_normal_(layer.weifght, a = 0, mode = "fan_in", nonlinearity='leaky_relu', generator=None)

        return nn.Sequential(*layers)


    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.covn2_x(x)
        x = self.covn3_x(x)
        x = self.covn4_x(x)
        x = self.covn5_x(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.classifier(x)
        return x