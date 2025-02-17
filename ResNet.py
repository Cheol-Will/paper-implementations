import torch
import torch.nn as nn
import torch.functional as F

class LeNet(nn.Module):

    def __init__(self):
        super(LeNet, self).__init__()

        self.conv1 = nn.Conv2d(1, 6, 5) # in_channels, out_channels, kernel_size
        self.conv2 = nn.Conv2d(6, 16, 3)
        self.fc1 = nn.Linear(16 * 6 * 6, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10) # 100 class labels

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
    def num_flat_features(self, x):
        size = x.size()[1:] # exclude batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
    
class VGGBlock(nn.Module):

    def __init__(self, in_channels, out_channels, stride):
        super(VGGBlock, self).__init__()
        self.conv1 = nn.Cond2d(in_channels, out_channels, 3, stride = stride, padding = 1)
        self.bn1 = nn.BatchNorm2d(out_channels)
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        return out

class VGG_34(nn.Module):

    def __init__(self):
        super(VGG_34, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(3, 64, 7, stride = 2)
        self.bn1 = nn.BatchNorm2d(64)
        self.vggblock1 = self.make_layer(64, 6*2, 1)
        self.vggblock2 = self.make_layer(128, 8*2, 2)
        self.vggblock3 = self.make_layer(256, 12*2, 2)
        self.vggblock4 = self.make_layer(512, 6*2, 2)
        self.classifier = nn.Sequential(
            nn.Linear(512 * 1 * 1, 1000),
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

        x = F.avg_pool2d(x, 7)
        x = self.classifier(x)
        return x
    
class ResBlock(nn.Module):
    
    def __init__(self, in_channels, out_channels, stride):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Cond2d(in_channels, out_channels, 3, stride = stride, padding = 1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Cond2d(out_channels, out_channels, 3, stride = stride, padding = 1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential([
                nn.Cond2d(in_channels, out_channels, 1, stride = 2),
                nn.BatchNorm2d(out_channels)
            ])

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.covn2(x))
        out = self.shortcut(x)
        out = F.relu(out)
        return out
    
class ResBottleNeck(nn.Module):
    def __init__(self):
        pass

    def forward(self, x):
        out = x
        return x



class ResNet(nn.Module):

    def __init__(self, block, in_channels=64):
        super(ResNet, self).__init__()

        self.in_channels = in_channels
        self.conv1 = nn.Conv2d(3, 64, 7, stride = 2)
        self.bn1 = nn.BatchNorm2d(64)
        self.resblock1 = self.make_layer(block, 64, 6, 1)
        self.resblock2 = self.make_layer(block, 128, 8, 2)
        self.resblock3 = self.make_layer(block, 256, 12, 2)
        self.resblock4 = self.make_layer(block, 512, 6, 2)
        self.classifier = nn.Sequential(
            nn.Linear(512 * 1 * 1, 1000) 
        )

    def make_layer(self, block, out_channels, num_blocks, first_stride):
        stride_list = [first_stride] + [1] * num_blocks
        layers = []
        for s in stride_list:
            layers.append(block(self.in_channels, out_channels, s))
            self.in_channels = out_channels

        return nn.Sequential(*layers)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x))) 
        x = F.avg_pool2d(x, 2)
        x = self.resblock1(x)
        x = self.resblock2(x)
        x = self.resblock3(x)
        x = self.resblock4(x)
        x = F.avg_pool2d(x, 7)
        x = self.classifier(x)
        return x

