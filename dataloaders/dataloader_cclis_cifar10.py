
import copy
import numpy as np

import torch
from torchvision import transforms, datasets
from torch.utils.data import Subset, Dataset
from torch.utils.data.dataset import ConcatDataset



class IS_Subset(Dataset):
    def __init__(self, dataset, indices, IS_weight):
        self.dataset = dataset
        self.indices = indices
        self.weight = IS_weight

    def __getitem__(self, idx):
        index = self.indices[idx]
        weight = self.weight[idx]
        image, label = self.dataset[index]

        # print("[DEBUG] __getitem__ called")  # ← 絶対呼ばれるはず
        return image, label, weight, index

    def __len__(self):
        return len(self.indices)


class TwoCropTransform:
    """Create two crops of the same image"""
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        return [self.transform(x), self.transform(x)]


#=============================
# CIFAR10 データローダー
#=============================
def set_loader_cclis_cifar10(cfg, normalize, trainer, replay_indices, training=True):

    importance_weight = trainer.importance_weight


    # 値の定義
    size = cfg.dataset.size
    target_task = cfg.continual.target_task
    cls_per_task = cfg.continual.cls_per_task
    batch_size = cfg.train.batch_size
    num_workers = cfg.num_workers

    # データ拡張の定義
    train_transform = transforms.Compose([
        transforms.Resize(size=(size, size)),
        transforms.RandomResizedCrop(size=size, scale=(0.1 if cfg.dataset.name=='tiny-imagenet' else 0.2, 1.)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=size//20*2+1, sigma=(0.1, 2.0))], p=0.5 if size>32 else 0.0),
        transforms.ToTensor(),
        normalize,
    ])


    # 現在タスクのクラス
    target_classes = list(range(target_task*cls_per_task, (target_task+1)*cls_per_task))
    print(target_classes)

    # 現在タスクのクラスのみを対象にインデックスを取り出す
    subset_indices = []
    subset_importance_weight = []

    _train_dataset = datasets.CIFAR10(root=cfg.dataset.folder,
                                      transform=TwoCropTransform(train_transform),
                                      download=True)
    

    for tc in target_classes:
        target_class_indices = np.where(np.array(_train_dataset.targets) == tc)[0]
        subset_indices += np.where(np.array(_train_dataset.targets) == tc)[0].tolist()  # cur_sample index, list
        tc_num = (np.array(_train_dataset.targets) == tc).sum()
        
        subset_importance_weight += list(np.ones(tc_num) / tc_num)  # cur_sample importance weight, list

    _subset_indices, _subset_importance_weight = copy.deepcopy(subset_indices), copy.deepcopy(subset_importance_weight)


    if len(replay_indices) > 0 and training:
        prev_dataset = IS_Subset(_train_dataset, replay_indices, importance_weight)
        cur_dataset = IS_Subset(_train_dataset, _subset_indices, _subset_importance_weight)

        dataset_len_list = [len(prev_dataset), len(cur_dataset)]

        train_dataset = ConcatDataset([prev_dataset, cur_dataset])


    else:
        _subset_indices += replay_indices
        _subset_importance_weight += importance_weight

        train_dataset = IS_Subset(_train_dataset, _subset_indices, _subset_importance_weight)


