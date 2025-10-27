# python
import os
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms


class ImageFolderDataModule:
    """
    用于从本地文件夹以 ImageFolder 方式加载训练/验证数据的封装类。

    参数:
      base_data_path: 根数据目录（例如: `CIFAR10`）
      train_subdir: 训练子目录相对于 base_data_path 的路径
      val_subdir: 验证子目录相对于 base_data_path 的路径
      batch_size: batch 大小
      num_workers: DataLoader 的 num_workers
      shuffle_train: 是否对训练集进行 shuffle
    """
    def __init__(
        self,
        base_data_path,
        train_subdir,
        val_subdir,
        batch_size=128,
        num_workers=2,
        shuffle_train=True
    ):
        self.base_data_path = base_data_path
        self.train_subdir = train_subdir
        self.val_subdir = val_subdir

        cifar10_mean = (0.4914, 0.4822, 0.4465)
        cifar10_std = (0.2470, 0.2435, 0.2616)
        self.transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(cifar10_mean, cifar10_std),
        ])
        self.transform_val = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(cifar10_mean, cifar10_std),
        ])
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle_train = shuffle_train

        self.train_dataset = None
        self.val_dataset = None
        self.train_loader = None
        self.val_loader = None

    def setup(self):
        """创建 torchvision.datasets.ImageFolder 和对应的 DataLoader（延迟初始化）"""
        train_root = os.path.join(self.base_data_path, self.train_subdir)
        val_root = os.path.join(self.base_data_path, self.val_subdir)

        if not os.path.isdir(train_root):
            raise FileNotFoundError(f"Training path not found: `{train_root}`")
        if not os.path.isdir(val_root):
            raise FileNotFoundError(f"Validation path not found: `{val_root}`")

        self.train_dataset = torchvision.datasets.ImageFolder(root=train_root, transform=self.transform_train)
        self.val_dataset = torchvision.datasets.ImageFolder(root=val_root, transform=self.transform_val)

        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle_train,
            num_workers=self.num_workers
        )
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers
        )

    def get_loaders(self):
        """返回 (train_loader, val_loader)。如果尚未 setup，会先调用 setup()."""
        if self.train_loader is None or self.val_loader is None:
            self.setup()
        return self.train_loader, self.val_loader

    @property
    def classes(self):
        """返回训练集的类别列表（如果尚未初始化则触发 setup）"""
        if self.train_dataset is None:
            self.setup()
        return self.train_dataset.classes
