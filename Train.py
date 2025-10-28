import torch
from torch import nn

from ViT import VisionTransformer
from DyViT import DynamicVisionTransformer
from AViT import AdaptiveVisionTransformer
import os
import time
import csv
from typing import Dict, List
import torch.optim as optim
from DataSet import ImageFolderDataModule  # 导入自定义的数据加载模块
import matplotlib.pyplot as plt

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
    return val_loss, val_acc

def train_model(model, train_loader, val_loader, criterion, optimizer, device, epochs):
    """
    训练循环封装函数，返回包含训练/验证损失与精度的历史记录字典。
    """
    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "train_time": [],
        "val_time": []
    }
    for epoch in range(epochs):
        print(f"\n--- Epoch {epoch + 1}/{epochs} ---")
        train_start = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        train_end = time.time()
        print(f'Epoch {epoch + 1} Summary: Train Loss: {train_loss:.3f} | Train Acc: {train_acc:.3f}% | Train Time: {train_end - train_start:.2f}s')
        val_loss, val_acc = validate_model(model, val_loader, criterion, device)
        val_end = time.time()
        print(f'Validation Loss: {val_loss:.3f} | Validation Acc: {val_acc:.3f}% | Validation Time: {val_end - train_end:.2f}s')
        scheduler.step()
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["train_time"].append(train_end - train_start)
        history["val_time"].append(val_end - train_end)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
    return history

def visualize(history, path='loss.png'):
    """
    可视化训练和验证的损失与精度曲线。
    """
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(epochs, history["train_loss"], label='Train Loss')
    axes[0].plot(epochs, history["val_loss"], label='Validation Loss')
    axes[0].set_xlabel('Epochs')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()

    axes[1].plot(epochs, history["train_acc"], label='Train Accuracy')
    axes[1].plot(epochs, history["val_acc"], label='Validation Accuracy')
    axes[1].set_xlabel('Epochs')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].set_title('Training and Validation Accuracy')
    axes[1].legend()

    fig.tight_layout()

    # 确保目录存在
    dirpath = os.path.dirname(path)
    if dirpath:
        os.makedirs(dirpath, exist_ok=True)

    # 使用 fig.savefig 保存（在 show 之前），并关闭 figure
    fig.savefig(path, bbox_inches='tight', dpi=300)
    plt.show()
    plt.close(fig)

def save_history_csv(history: Dict[str, List[float]], path: str):
    """保存为 CSV，每行对应一个 epoch。"""
    keys = ["train_loss", "train_acc", "val_loss", "val_acc"]
    length = len(history.get(keys[0], []))
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch"] + keys)
        for i in range(length):
            row = [i + 1] + [history[k][i] for k in keys]
            writer.writerow(row)

def init_vit_weights(m):
    """对模块 m 进行常用初始化（供 model.apply 使用）。"""
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LayerNorm):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)

def initialize_vit_model(model, std=0.02):
    """
    1) 对模块参数应用常规模块初始化（model.apply）。
    2) 单独初始化可学习的位置嵌入与 cls token（模型属性，必须显式处理）。
    """
    model.apply(init_vit_weights)

    # 使用截断正态（如果可用）或普通正态作为替代
    try:
        nn.init.trunc_normal_(model.pos_embed, std=std)
        nn.init.trunc_normal_(model.cls_token, std=std)
    except AttributeError:
        nn.init.normal_(model.pos_embed, mean=0.0, std=std)
        nn.init.normal_(model.cls_token, mean=0.0, std=std)
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
    EPOCHS = 100
    BATCH_SIZE = 128
    LEARNING_RATE = 1e-4
    KEEP_RATE = 0.7 # 设置每个动态注意力层的 token 保留率
    MODEL_TYPE = 'AViT'

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # --- 数据加载和增强 ---
    print("Loading data from local folders...")
    # 使用自定义的数据模块加载数据
    data_module = ImageFolderDataModule(
        base_data_path='/home/woshihg/CIFAR10',
        train_subdir='CIFAR10_imbalanced/CIFAR10_unbalance',
        val_subdir='CIFAR10_balanced/CIFAR10_balance',
        batch_size=BATCH_SIZE,
        num_workers=8,
        shuffle_train=True
    )
    data_module.setup()
    train_loader = data_module.train_loader
    val_loader = data_module.val_loader

    # --- 计算类别权重以解决样本不平衡问题 ---
    print("Calculating class weights for imbalanced dataset...")
    # 1. 获取训练集中每个类别的样本数
    class_counts = torch.bincount(torch.tensor(data_module.train_dataset.targets))
    print(f"Class counts: {class_counts.tolist()}")

    # 2. 计算权重 (公式: 总样本数 / (类别数 * 每个类别的样本数))
    total_samples = sum(class_counts)
    num_classes = len(class_counts)
    class_weights = total_samples / (num_classes * class_counts.float())
    class_weights = class_weights.to(device) # 将权重移动到正确的设备

    # --- 初始化模型、损失函数和优化器 (与之前相同) ---
    print("Initializing model...")
    if MODEL_TYPE == 'AViT':
        model = AdaptiveVisionTransformer(
            img_size=IMG_SIZE,
            patch_size=PATCH_SIZE,
            in_channels=3,
            embed_dim=EMBED_DIM,
            num_layers=NUM_LAYERS,
            num_heads=NUM_HEADS,
            mlp_dim=MLP_DIM,
            num_classes=NUM_CLASSES,
            dropout=DROPOUT,
            keep_rate=KEEP_RATE # 传入保留率
        ).to(device)
    elif MODEL_TYPE == 'DyViT':
        model = DynamicVisionTransformer(
            img_size=IMG_SIZE,
            patch_size=PATCH_SIZE,
            in_channels=3,
            embed_dim=EMBED_DIM,
            num_layers=NUM_LAYERS,
            num_heads=NUM_HEADS,
            mlp_dim=MLP_DIM,
            num_classes=NUM_CLASSES,
            dropout=DROPOUT,
            keep_rate=KEEP_RATE
        ).to(device)
    else:
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


    # 计算模型参数数量
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model initialized with {num_params / 1e6:.2f} million parameters.")
    # 计算FLOPs
    from thop import profile
    dummy_input = torch.randn(1, 3, IMG_SIZE, IMG_SIZE).to(device)
    flops, params = profile(model, inputs=(dummy_input,), verbose=False)
    print(f"FLOPs: {flops / 1e9:.6f} GFLOPs")

    init_vit_weights(model)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    # --- 训练循环 ---
    print("Starting training...")
    history = train_model(model, train_loader, val_loader, criterion, optimizer, device, EPOCHS)
    print("Training finished!")
    # 计算平均每 epoch 时间
    avg_train_time = sum(history["train_time"]) / EPOCHS
    avg_val_time = sum(history["val_time"]) / EPOCHS
    print(f"Average training time per epoch: {avg_train_time:.2f}s")
    print(f"Average validation time per epoch: {avg_val_time:.2f}s")
    # 存储数据表history到本地
    save_history_csv(history, "history.csv")

    visualize(history, path='vit_training_history.png')
