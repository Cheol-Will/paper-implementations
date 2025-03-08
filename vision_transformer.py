import torch
import torch.nn as nn
import torch.nn.functional as F

from torchsummary import summary

class Encoder(nn.Module):
    def __init__(self, hidden_dim, num_heads, mlp_dim):
        super(Encoder, self).__init__()
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.multi_head_attention = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True) # input shape: (batch_size, seq_length, hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, hidden_dim)
        )

    def forward(self, x):
        out = self.ln1(x)
        # attn_output, attn_output_weights = self.multi_head_attention(out, out, out)
        out = x + self.multi_head_attention(out, out, out)[0]
        x = self.mlp(self.ln2(x))

        return x

class VisionTransformer(nn.Module):
    def __init__(self, image_size, hidden_dim, num_heads, mlp_dim, patch_size, depth, num_classes):
        super(VisionTransformer, self).__init__()
        height, width = image_size

        self.patch_embedding = nn.Conv2d(3, hidden_dim, patch_size, patch_size)
        self.pos_embedding = nn.init.normal_(nn.Parameter(torch.empty(1, height//patch_size * width//patch_size, hidden_dim)))
        self.transformer_blocks = nn.Sequential(
            *[Encoder(hidden_dim, num_heads, mlp_dim) for _ in range(depth)]
        )
        self.clf = nn.Linear(hidden_dim, 10)

    def forward(self, x):
        x = self.patch_embedding(x)
        batch_size, hidden_dim, height, width = x.shape
        x = x.reshape(batch_size, height * width, hidden_dim)
        x += self.pos_embedding
        x = self.transformer_blocks(x)
        x = x.mean(axis = 1)
        x = self.clf(x)

        return x