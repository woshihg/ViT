import torch
import torch.nn as nn


class PatchEmbedding(nn.Module):
    """
    将图像分割成块（Patches）并进行线性嵌入。

    参数:
        img_size (int): 输入图像的尺寸（假设为方形）。
        patch_size (int): 每个 patch 的尺寸（假设为方形）。
        in_channels (int): 输入图像的通道数（例如 3 对应 RGB）。
        embed_dim (int): 嵌入后的维度（Transformer 的隐藏层大小）。
    """

    def __init__(self, img_size, patch_size, in_channels, embed_dim):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2

        # 使用一个卷积层来实现 patch 切割和嵌入
        # kernel_size 和 stride 都等于 patch_size
        self.proj = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

    def forward(self, x):
        # x 形状: (B, C, H, W) -> (B, embed_dim, H_patch, W_patch)
        x = self.proj(x)

        # 展平: (B, embed_dim, H_patch, W_patch) -> (B, embed_dim, N_patches)
        x = x.flatten(2)

        # 维度换位: (B, embed_dim, N_patches) -> (B, N_patches, embed_dim)
        x = x.transpose(1, 2)
        return x


class VisionTransformer(nn.Module):
    """
    Vision Transformer (ViT) 模型。

    参数:
        img_size (int): 输入图像尺寸。
        patch_size (int): Patch 尺寸。
        in_channels (int): 输入通道数。
        embed_dim (int): 嵌入维度 (D)。
        num_layers (int): Transformer Encoder 层数。
        num_heads (int): Transformer Encoder 中的多头注意力头数。
        mlp_dim (int): Transformer Encoder 中 MLP 层的隐藏维度。
        num_classes (int): 最终分类的类别数。
        dropout (float): Dropout 比例。
    """

    def __init__(self,
                 img_size=224,
                 patch_size=16,
                 in_channels=3,
                 embed_dim=768,
                 num_layers=12,
                 num_heads=12,
                 mlp_dim=3072,
                 num_classes=1000,
                 dropout=0.1):
        super().__init__()

        # 1. Patch 嵌入
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        num_patches = self.patch_embed.n_patches

        # 2. [CLS] token
        # (1, 1, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # 3. 位置编码 (Positional Embedding)
        # 需要为 (CLS token + 所有 patches) 编码
        # (1, num_patches + 1, embed_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=dropout)

        # 4. Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=mlp_dim,
            dropout=dropout,
            activation="gelu",  # ViT 通常使用 GELU
            batch_first=True  # (B, Seq, Dim)
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers
        )

        # 5. 分类头 (MLP Head)
        self.norm = nn.LayerNorm(embed_dim)  # 最终的 LayerNorm
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        # x 初始形状: (B, C, H, W)
        B = x.shape[0]

        # 1. Patch 嵌入
        # (B, C, H, W) -> (B, num_patches, embed_dim)
        x = self.patch_embed(x)

        # 2. 添加 [CLS] token
        # (1, 1, embed_dim) -> (B, 1, embed_dim)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        # (B, 1, embed_dim) + (B, num_patches, embed_dim) -> (B, num_patches + 1, embed_dim)
        x = torch.cat((cls_tokens, x), dim=1)

        # 3. 添加位置编码
        x = x + self.pos_embed
        x = self.pos_drop(x)

        # 4. 通过 Transformer Encoder
        # (B, num_patches + 1, embed_dim) -> (B, num_patches + 1, embed_dim)
        x = self.transformer_encoder(x)

        # 5. 提取 [CLS] token 用于分类
        # (B, num_patches + 1, embed_dim) -> (B, embed_dim)
        cls_output = x[:, 0]

        # 6. 通过分类头
        cls_output = self.norm(cls_output)
        logits = self.head(cls_output)

        return logits


# --- 使用示例 ---

if __name__ == '__main__':
    # 模拟一个 ViT-Base (ViT-B/16) 的配置
    vit_b16 = VisionTransformer(
        img_size=224,
        patch_size=16,
        embed_dim=768,
        num_layers=12,
        num_heads=12,
        mlp_dim=3072,
        num_classes=1000  # 假设是 ImageNet
    )

    # 创建一个模拟的输入 (Batch=4, Channels=3, H=224, W=224)
    dummy_image = torch.randn(4, 3, 224, 224)

    # 前向传播
    logits = vit_b16(dummy_image)

    print(f"输入形状: {dummy_image.shape}")
    print(f"输出 Logits 形状: {logits.shape}")  # 应该是 (4, 1000)