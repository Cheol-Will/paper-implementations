import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvMixerBlock(nn.Module):
    def __init__(self, hidden_dim, kernel_size):
        super(ConvMixerBlock, self).__init__()
        self.depthwise = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size, groups=hidden_dim, padding="same"),
            nn.GELU(),
            nn.BatchNorm2d(hidden_dim)
        )

        self.pointwise = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 1),
            nn.GELU(),
            nn.BatchNorm2d(hidden_dim)
        )
        
    def forward(self, x):
        out = x + self.depthwise(x)
        out = self.pointwise(out)
        return out
    
class ConvMixer(nn.Module):
    def __init__(self, hidden_dim, depth, patch_size, kernel_size, num_classes):
        super(ConvMixer, self).__init__()
        """
            hidden_dim: dimension of embedding
            depth: total network depth
            patch_size: the resolution of patch (eg. p x p)
            kernel_size: kernel size of depthwise convolution
        """

        self.hidden_dim = hidden_dim
        self.depth = depth
        self.patch_size = patch_size
        self.kernel_size = kernel_size

        # patchify
        self.patch = nn.Sequential(
            nn.Conv2d(3, hidden_dim, patch_size, patch_size),
            nn.GELU(),
            nn.BatchNorm2d(hidden_dim)
        )
        
        # ConvMixer Block Groups 
        self.convmixer = nn.Sequential(
            *[ConvMixerBlock(hidden_dim, kernel_size) for _ in range(depth)]
        )
        
        # Classification Layer
        self.clf = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(hidden_dim, num_classes),
        )


    def forward(self, x):
        out = self.patch(x)
        out = self.convmixer(out)
        out = self.clf(out)
        return out
