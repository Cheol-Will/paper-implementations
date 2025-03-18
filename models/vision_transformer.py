import torch
import torch.nn as nn
import torch.nn.functional as F

from torchsummary import summary

class VisionTransformerBlock(nn.Module):
    def __init__(self, hidden_dim, num_heads, mlp_dim):
        super(VisionTransformerBlock, self).__init__()
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.multi_head_attention = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True) # input shape: (batch_size, seq_length, hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, hidden_dim)
        )

    def forward(self, x):
        # attn_output, attn_output_weights = self.multi_head_attention(out, out, out)
        # out = x + self.multi_head_attention(out, out, out)[0]
        out = self.ln1(x)
        out, _ = x + self.multi_head_attention(out, out, out)
        out = out + self.mlp(self.ln2(out))

        return out

class VisionTransformer(nn.Module):
    def __init__(self, image_size, hidden_dim, num_heads, mlp_dim, patch_size, depth, num_classes):
        super(VisionTransformer, self).__init__()
        height, width = image_size

        self.patch_embedding = nn.Conv2d(3, hidden_dim, patch_size, patch_size)
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim))
        self.pos_embedding = nn.Parameter(nn.init.normal_(torch.empty(1, height//patch_size * width//patch_size + 1, hidden_dim)))
        self.transformer_blocks = nn.Sequential(
            *[VisionTransformerBlock(hidden_dim, num_heads, mlp_dim) for _ in range(depth)]
        )
        self.clf = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        x = self.patch_embedding(x)
        batch_size, hidden_dim, height, width = x.shape
        x = x.reshape(batch_size, hidden_dim, height * width)
        x = x.transpose(1, 2) # transpose so that shape(and values) becomes (batch_size, seq_length, dim_hidden)
        
        # concat patch_embedding with class token
        cls_token_batched = self.cls_token.expand(batch_size, -1, -1)
        x = torch.concat([cls_token_batched, x], dim=1)
        x += self.pos_embedding
        x = self.transformer_blocks(x)

        # use only class token for prediction
        x = x[:, 0, :]
        x = self.clf(x)

        return x