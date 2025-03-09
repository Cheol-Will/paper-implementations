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
    def __init__(self, hidden_dim, patch_size, c_hidden, s_hidden, depth, height, width, num_classes):
        super(MLPMixer, self).__init__()
        self.patch = nn.Conv2d(3, hidden_dim, patch_size, patch_size)
        self.mixer = nn.Sequential(
            *[MixerBlock(hidden_dim, height//patch_size * width//patch_size, c_hidden, s_hidden) for _ in range(depth)]
        )
        self.ln = nn.LayerNorm(hidden_dim)
        self.clf = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = self.patch(x)
        x = x.flatten(2).transpose(1, 2) # (batch_size, hidden_dim, height, width) -> (batch_size, seq_length, hidden_dim)
        x = self.mixer(x)
        x = self.ln(x)
        x = x.mean(axis = 1)
        x = self.clf(x)

        return x