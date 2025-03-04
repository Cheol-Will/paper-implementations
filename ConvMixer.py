import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvMixerBlock(nn.Module):
    def __init__(self, h, k):
        super(ConvMixerBlock, self).__init__()
        self.depthwise = nn.Sequential(
            nn.Conv2d(h, h, k, groups = h, padding = "same"),
            nn.GELU(),
            nn.BatchNorm2d(h)
        )

        self.pointwise = nn.Sequential(
            nn.Conv2d(h, h, 1),
            nn.GELU(),
            nn.BatchNorm2d(h)
        )
        
    def forward(self, x):
        out = x + self.depthwise(x)
        out = self.pointwise(out)
        return out
    
class ConvMixer(nn.Module):
    def __init__(self, h, d, p, k, num_class):
        super(ConvMixer, self).__init__()
        """
            h: dimension of embedding
            d: total network depth
            p: the resolution of patch (eg. p x p)
            k: kernel size of depthwise convolution
        """

        self.dim_embedding = h
        self.depth = d
        self.patch_size = p
        self.kernel_size = k

        # patchify
        self.patch = nn.Sequential(
            nn.Conv2d(3, h, p, stride = p),
            nn.GELU(),
            nn.BatchNorm2d(h)
        )
        
        # ConvMixer Block Groups 
        self.convmixer = nn.Sequential(
            *[ConvMixerBlock(h, k) for _ in range(d)]
        )
        
        # Classification Layer
        self.clf = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(h, num_class),
            nn.LogSoftmax(dim = 1) 
        )


    def forward(self, x):
        out = self.patch(x)
        out = self.convmixer(out)
        out = self.clf(out)
        return out
