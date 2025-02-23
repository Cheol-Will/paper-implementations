import torch
import torch.nn as nn
import torch.nn.functional as F




class StDepthBlock(nn.Module):
   
    def __init__(self, in_channels, out_channels, survival_probabily):
        super(StDepthBlock, self).__init__()
        self.res = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3 ,1, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )
        self.shortcut = nn.Sequential()

        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                
            )

    def forward(self, x):
        if torch.bernoulli(self.survival_probabily) == 1:
            x += self.res(x)
        return x
    
class StDepth(nn.Module):
    """
        3 groups with 18 residual blocks each. 
        # of filters in each group are 16, 32, 64
    """

    def __init__(self, block, in_channels, num_blocks_list):
        super(StDepth, self).__init__()
        self.in_channels = in_channels
        self.st1 = self.make_layer(block, 16, num_blocks_list[0])
        self.tran1 = nn.Sequential(
            nn.AvgPool2d()
        )
        # transition layer

        self.st2 = self.make_layer(block, 32, num_blocks_list[1])
        
        # transition layer
        
        self.st3 = self.make_layer(block, 16, num_blocks_list[2])


    def make_layer(self, block, out_channels, num_blocks):
        layers = []
        for num in range(num_blocks):
            layers.append(block(self.in_channels, out_channels, num))            
            self.in_channels = out_channels

        return nn.Sequential(*layers)

    def forward(self, x):

        return x
        

