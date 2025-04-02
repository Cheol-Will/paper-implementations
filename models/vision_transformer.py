import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttention(nn.Module):
    def __init__(self, hidden_dim, num_heads):
        super(SelfAttention, self).__init__()
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        self.proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, query, key, value):
        Q = self.query(query)
        K = self.key(key).transpose(1, 2) # (batch, seq, dim) -> (batch, dim, seq)
        V = self.value(value)

        # softmax along seq dimension and normalize
        # K Transpose 반영 필요
        A = F.softmax(torch.einsum("bij,bjk->bik", Q, K), dim=1) / (hidden_dim**(1/2))
        V = torch.einsum("bij,bjk->bik", A, V)
        out = self.proj(V)
        return out, A

class MultiheadAttention(nn.Module):
    def __init__(self, hidden_dim, num_heads, batch_first=True):
        super(MultiheadAttention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads 
        self._query = nn.Linear(hidden_dim, hidden_dim)
        self._key = nn.Linear(hidden_dim, hidden_dim)
        self._value = nn.Linear(hidden_dim, hidden_dim)  
        self.proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, query, key, value):
        # input shape: (batch_size, seq_length, hidden_dim)
        # X: (B, N, D)
        # Q, K, V: (B, N, D) -> (B, N, K, D/K)

        B, N, D = query.shape
        H = self.num_heads
        head_dim = self.head_dim

        # (B, N, D) -> (B, N, H, D') -> (B, H, N, D')
        Q = self._query(query).reshape(B, N, H, head_dim).transpose(1,2)
        K = self._key(key).reshape(B, N, H, head_dim).transpose(1,2)
        V = self._value(value).reshape(B, N, H, head_dim).transpose(1,2)

        # (batch, head, seq, dim)
        A = F.softmax(torch.einsum("bhij,bhkj->bhik", Q, K) / (head_dim**(0.5)), dim=-1) # (B, H, N, N)
        V_ = torch.einsum("bhij,bhjk->bhik", A, V) # (B, H, N, D')
        V_ = V_.transpose(1, 2).reshape(B, N, D) # (B, N, H, D') -> (B, N, D)
        out = self.proj(V_)

        return out, A


class VisionTransformerBlock(nn.Module):
    def __init__(self, hidden_dim, num_heads, mlp_dim):
        super(VisionTransformerBlock, self).__init__()
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.multi_head_attention = MultiheadAttention(hidden_dim, num_heads, batch_first=True) # input shape: (batch_size, seq_length, hidden_dim)
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
        out, _ = self.multi_head_attention(out, out, out)
        out = x + out
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


# model = VisionTransformer(image_size=(32, 32), hidden_dim=384, num_heads=6, mlp_dim=64*4, patch_size=4, depth=8, num_classes=10)
# print(count_parameters(model))