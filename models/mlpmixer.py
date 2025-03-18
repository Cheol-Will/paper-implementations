import torch 
import torch.nn as nn
import torch.nn.functional as F

class MixerBlock(nn.Module):
    def __init__(self, hidden_dim, seq_length, c_hidden, s_hidden):
        super(MixerBlock, self).__init__()
        self.ln = nn.LayerNorm(hidden_dim)
        self.token = nn.Sequential(
            nn.Linear(seq_length, c_hidden),
            nn.GELU(),
            nn.Linear(c_hidden, seq_length)
        )
        self.channel = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, s_hidden),
            nn.GELU(),
            nn.Linear(s_hidden, hidden_dim)
        )

    def forward(self, x):
        out = self.ln(x)
        out = self.token(out.transpose(1, 2))
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
        self.clf = nn.Sequential(nn.Linear(hidden_dim, num_classes))

    def forward(self, x):
        x = self.patch(x)
        x = x.flatten(2).transpose(1, 2) # (batch_size, hidden_dim, height, width) -> (batch_size, seq_length, hidden_dim)
        x = self.mixer(x)
        x = self.ln(x)
        x = x.mean(axis = 1)
        x = self.clf(x)

        return x