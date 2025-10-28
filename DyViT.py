import torch
import torch.nn as nn
from ViT import PatchEmbedding

class DynamicAttention(nn.Module):
    """
    一个包含 token 修剪机制的 Transformer Encoder 层。
    """
    def __init__(self, embed_dim, num_heads, mlp_dim, dropout, keep_rate=0.7):
        super().__init__()
        self.keep_rate = keep_rate
        self.layer_norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.layer_norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, embed_dim),
            nn.Dropout(dropout)
        )
        # 决策网络，用于为每个 token 生成一个分数
        self.pruning_predictor = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 4),
            nn.GELU(),
            nn.Linear(embed_dim // 4, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # 1. 标准 Transformer 操作
        x = self.layer_norm1(x)
        attn_output, _ = self.attn(x, x, x)
        x = x + attn_output
        x = x + self.mlp(self.layer_norm2(x))

        # 2. Token 修剪
        if not self.training or x.shape[1] <= 2: # 如果不是训练模式或 token 数量太少，则不修剪
            return x

        cls_token = x[:, :1, :]
        patch_tokens = x[:, 1:, :]

        # 3. 计算决策分数并决定保留哪些 token
        scores = self.pruning_predictor(patch_tokens) # (B, N_patches, 1)
        num_keep = int(self.keep_rate * patch_tokens.shape[1])

        if num_keep > 0:
            # 选取分数最高的 token 的索引
            keep_indices = scores.squeeze(-1).topk(num_keep, dim=1).indices
            # 根据索引筛选 token
            kept_tokens = torch.gather(patch_tokens, 1, keep_indices.unsqueeze(-1).expand(-1, -1, patch_tokens.shape[-1]))
            # 重新组合 [CLS] token 和保留下来的 patch token
            x = torch.cat((cls_token, kept_tokens), dim=1)

        else: # 如果需要保留的 token 数量为0，则只保留 cls_token
            x = cls_token

        return x


class DynamicVisionTransformer(nn.Module):
    """
    使用动态 Token 修剪的 Vision Transformer (DynamicViT)。
    """
    def __init__(self,
                 img_size,
                 patch_size,
                 in_channels,
                 embed_dim,
                 num_layers,
                 num_heads,
                 mlp_dim,
                 num_classes,
                 dropout=0.1,
                 keep_rate=0.7): # 新增 keep_rate 参数
        super().__init__()

        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        num_patches = self.patch_embed.n_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=dropout)

        # 创建一个包含多个 DynamicAttention 层的列表
        self.transformer_layers = nn.ModuleList([
            DynamicAttention(embed_dim, num_heads, mlp_dim, dropout, keep_rate)
            for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # 只为当前存在的 token 添加位置编码
        x = x + self.pos_embed[:, :x.size(1), :]
        x = self.pos_drop(x)

        # 逐层通过 DynamicAttention
        for layer in self.transformer_layers:
            x = layer(x)

        # 只取 [CLS] token 用于分类
        cls_output = x[:, 0]
        cls_output = self.norm(cls_output)
        logits = self.head(cls_output)

        return logits


