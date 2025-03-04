import torch 
import torch.nn as nn
import torch.nn.functional as F

class MixerBlock(nn.Module):
    def __init__(self, c, s, c_hidden, s_hidden):
        super(MixerBlock, self).__init__()
        self.token = nn.Sequential(
            nn.LayerNorm(s),
            nn.Linear(s, c_hidden),
            nn.GELU(),
            nn.Linear(c_hidden, s)
        )
        self.channel = nn.Sequential(
            nn.LayerNorm(c),
            nn.Linear(c, s_hidden),
            nn.GELU(),
            nn.Linear(s_hidden, c)
        )

    def forward(self, x):
        out = self.token(x.transpose(1, 2))
        out = out.transpose(1, 2)
        out = out + x  
        out = self.channel(out) + out

        return out

class MLPMixer(nn.Module):
    def __init__(self, c, p, c_hidden, s_hidden, h, w, num_class):
        super(MLPMixer, self).__init__()
        self.embedding_dimension = c
        self.patch = nn.Conv2d(3, c, p, stride = p)
        self.mixer = nn.Sequential(
            *[MixerBlock(c, h//p * w//p, c_hidden, s_hidden)]
        )
        self.clf = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(h//p * w//p, num_class),
            nn.LogSoftmax(dim = 1),
        )

    def forward(self, x):
        x = self.patch(x)
        x = x.flatten(2).transpose(1, 2) # S x C
        x = self.mixer(x)
        print(x.shape)
        x = self.clf(x)

        return x