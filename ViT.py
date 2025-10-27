import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import os  # 导入 os 模块来处理路径
from DataSet import ImageFolderDataModule  # 导入自定义的数据加载模块


# --- 1. ViT 模块代码 (与之前相同) ---

class PatchEmbedding(nn.Module):
    """
    将图像分割成块（Patches）并进行线性嵌入。
    """

    def __init__(self, img_size, patch_size, in_channels, embed_dim):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2

        self.proj = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

    def forward(self, x):
        x = self.proj(x)  # (B, embed_dim, H_patch, W_patch)
        x = x.flatten(2)  # (B, embed_dim, N_patches)
        x = x.transpose(1, 2)  # (B, N_patches, embed_dim)
        return x


class VisionTransformer(nn.Module):
    """
    Vision Transformer (ViT) 模型。
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
                 dropout=0.1):
        super().__init__()

        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        num_patches = self.patch_embed.n_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=mlp_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers
        )

        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        x = x + self.pos_embed
        x = self.pos_drop(x)

        x = self.transformer_encoder(x)

        cls_output = x[:, 0]
        cls_output = self.norm(cls_output)
        logits = self.head(cls_output)

        return logits


# --- 2. 训练和评估辅助函数 ---

def train_one_epoch(model, train_loader, criterion, optimizer, device):
    """训练一个 epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for i, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        if (i + 1) % 100 == 0:
            print(f'  [Batch {i + 1}/{len(train_loader)}] '
                  f'Train Loss: {running_loss / (i + 1):.3f} | '
                  f'Train Acc: {100. * correct / total:.3f}%')

    return running_loss / len(train_loader), 100. * correct / total


# --- 修改：将 test_model 重命名为 validate_model ---
def validate_model(model, val_loader, criterion, device):
    """在验证集上评估模型"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    val_loss = running_loss / len(val_loader)
    val_acc = 100. * correct / total
    # --- 修改：打印 "Validation" 而不是 "Test" ---
    print(f'Validation Loss: {val_loss:.3f} | Validation Acc: {val_acc:.3f}%')
    return val_loss, val_acc


# --- 3. 主执行函数 ---

if __name__ == '__main__':
    # --- 超参数设置 ---
    IMG_SIZE = 32
    PATCH_SIZE = 4
    NUM_CLASSES = 10
    EMBED_DIM = 512
    NUM_LAYERS = 6
    NUM_HEADS = 8
    MLP_DIM = 1024
    DROPOUT = 0.1
    EPOCHS = 20
    BATCH_SIZE = 128
    LEARNING_RATE = 1e-4

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # --- 数据加载和增强 ---
    print("Loading data from local folders...")
    # 使用自定义的数据模块加载数据
    data_module = ImageFolderDataModule(
        base_data_path='/home/woshihg/CIFAR10',
        train_subdir='CIFAR10_balanced/CIFAR10_balance',
        val_subdir='CIFAR10_imbalanced/CIFAR10_unbalance',
        batch_size=BATCH_SIZE,
        num_workers=2,
        shuffle_train=True
    )
    data_module.setup()
    train_loader = data_module.train_loader
    val_loader = data_module.val_loader

    # --- 初始化模型、损失函数和优化器 (与之前相同) ---
    print("Initializing model...")
    model = VisionTransformer(
        img_size=IMG_SIZE,
        patch_size=PATCH_SIZE,
        in_channels=3,
        embed_dim=EMBED_DIM,
        num_layers=NUM_LAYERS,
        num_heads=NUM_HEADS,
        mlp_dim=MLP_DIM,
        num_classes=NUM_CLASSES,
        dropout=DROPOUT
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    # --- 训练循环 ---
    print("Starting training...")
    for epoch in range(EPOCHS):
        print(f"\n--- Epoch {epoch + 1}/{EPOCHS} ---")

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        print(f'Epoch {epoch + 1} Summary: Train Loss: {train_loss:.3f} | Train Acc: {train_acc:.3f}%')

        # --- 修改：调用 validate_model 并使用 val_loader ---
        val_loss, val_acc = validate_model(
            model, val_loader, criterion, device
        )

        scheduler.step()

    print("Training finished!")