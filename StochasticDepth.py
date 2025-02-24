import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

class StDepthBlock(nn.Module):

    def __init__(self, in_channels, out_channels, stride,  survival_probabily, padding):
        super(StDepthBlock, self).__init__()
        self.survival_probabily = survival_probabily
        self.channel_pad = 0
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.res = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride = stride, padding = 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 3, stride = 1, padding = 1),
            nn.BatchNorm2d(out_channels),
        )
        self.shortcut = nn.Sequential()

        if padding and in_channels != out_channels:
            pad = (out_channels - in_channels)//2
            self.channel_pad = pad
            self.shortcut = nn.Sequential(
                nn.AvgPool2d(2, stride = 2),
                # nn.ZeroPad2d((0, 0, 0, 0, pad, pad)
            )

    def forward(self, x):
        out = self.shortcut(x)
        if self.channel_pad > 0:
            out = F.pad(out, (0, 0, 0, 0, self.channel_pad, self.channel_pad))
        if torch.bernoulli(self.survival_probabily) == 1:
            a = self.res(x)
            
            out += self.res(x)

        return F.relu(out)

class StDepth(nn.Module):
    """
        3 groups with 18 residual blocks each.
        # of filters in each group are 16, 32, 64
    """

    def __init__(self, block, num_blocks_list):
        super(StDepth, self).__init__()
        self.in_channels = 16

        self.conv1 = nn.Conv2d(3, 16, 7, stride = 1, padding = 3)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu1 = nn.ReLU()
        self.st1 = self.make_layer(block, 16, num_blocks_list[0], 1, False)
        self.st2 = self.make_layer(block, 32, num_blocks_list[1], 2, True)
        self.st3 = self.make_layer(block, 64, num_blocks_list[2], 2, True)

        self.clf = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  
            nn.Flatten(),
            nn.Linear(64 * 1 * 1, 1000),
            nn.LogSoftmax(dim = 1)
        )

    def make_layer(self, block, out_channels, num_blocks, first_stride, padding):

        stride_list = [first_stride] + [1] * (num_blocks-1)
        layers = []
        L = len(stride_list)
        for idx, stride in enumerate(stride_list):
            p = torch.tensor(1 - 0.5/L * idx)
            layers.append(block(self.in_channels, out_channels, stride, p, padding))
            self.in_channels = out_channels

        # weight initialization kaiming normal
        for layer in layers:
            if type(layer) == nn.Conv2d:
                nn.init.kaiming_normal_(layer.weight, a=0, mode='fan_in', nonlinearity='leaky_relu', generator=None)

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = F.avg_pool2d(x, 2)

        x = self.st1(x)
        x = self.st2(x)
        x = self.st3(x)
        print(f"shape before clf: {x.shape}")
        x = self.clf(x)
        return x



model = StDepth(StDepthBlock, [6, 6, 6])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
summary(model, input_size=(3, 256, 256), device=str(device))