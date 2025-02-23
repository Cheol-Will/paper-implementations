import torch
import torch.nn as nn
import torch.nn.functional as F
    
class ResBlock(nn.Module):
    
    def __init__(self, in_channels, out_channels, stride):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride = stride, padding = 1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride = 1, padding = 1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Sequential()

        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride = 2),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out
    
class ResBottleNeck(nn.Module):

    def __init__(self, in_channels, out_channels, stride):
        super(ResBottleNeck, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 1, stride = stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride = 1, padding = 1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels * 4, 1, stride = 1)
        self.bn3 = nn.BatchNorm2d(out_channels * 4)
        self.shortcut = nn.Sequential()

        if stride != 1 or in_channels != out_channels * 4:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels*4, 1, stride = stride),
                nn.BatchNorm2d(out_channels*4)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):

    def __init__(self, block, num_blocks_list, increase_rate = 1):
        super(ResNet, self).__init__()

        """
            num_blocks_list: the number of blocks for each block
            ResNet-18: [2, 2, 2, 2]
            ResNet-34/50: [3, 4, 6, 3]
            ResNet-101: [3, 4, 23, 3]
            ResNet-152: [3, 8, 36, 3]

            increare_rate: this rate determines output feature map channels in a block
            ResBlock, increase_rate = 1, 
            ResBottleNeck, increase_rate = 4
        """

        self.increase_rate = increase_rate
        self.in_channels = 64
        self.conv1 = nn.Conv2d(3, 64, 7, stride = 2)
        self.bn1 = nn.BatchNorm2d(64)
        self.covn2_x = self.make_layer(block, 64, num_blocks_list[0], 1)
        self.covn3_x = self.make_layer(block, 128, num_blocks_list[1], 2)
        self.covn4_x = self.make_layer(block, 256, num_blocks_list[2], 2)
        self.covn5_x = self.make_layer(block, 512, num_blocks_list[3], 2)

        # Navie softmax is not recommended since NLLLoss expects log to be computed between softmax and itself 
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(512 * 1 * 1 * increase_rate, 1000),
            nn.LogSoftmax(dim = 1) 
        )


    def make_layer(self, block, out_channels, num_blocks, first_stride):
        stride_list = [first_stride] + [1] * (num_blocks - 1)
        layers = []
        # create layers with given block (eg. ResBlock or ResBottleNeck)
        for stride in stride_list:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * self.increase_rate

        # weight initialization kaiming normal
        for layer in layers:
            if type(layer) == nn.Conv2d:
                nn.init.kaiming_normal_(layer.weight, a=0, mode='fan_in', nonlinearity='leaky_relu', generator=None)

        return nn.Sequential(*layers)
    

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x))) 
        x = F.avg_pool2d(x, 2)
        x = self.covn2_x(x)
        x = self.covn3_x(x)
        x = self.covn4_x(x)
        x = self.covn5_x(x)
        x = self.classifier(x)
        return x

# model = ResNet(ResBottleNeck, [2, 2, 2, 2], increase_rate = 4)
# model = ResNet(ResBlock, [2, 2, 2, 2], increase_rate = 1)

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model.to(device)  
# summary(model, input_size=(3, 32, 32), device=str(device))