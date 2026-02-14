import numpy as np

import torch
from torchvision import transforms, datasets
from torch.utils.data import Subset
from torch.utils.data import WeightedRandomSampler


def set_loader_cifar10_ncm_train(cfg, normalize, replay_indices):

    # 値の定義
    size = cfg.dataset.size
    target_task = cfg.linear.target_task
    cls_per_task = cfg.continual.cls_per_task
    batch_size = cfg.linear.train.batch_size
    num_workers = cfg.num_workers

    train_transform = transforms.Compose([
        transforms.Resize(size=(size, size)),
        transforms.ToTensor(),
        normalize,
    ])

    subset_indices = []
    _train_dataset = datasets.CIFAR10(root=cfg.dataset.folder,
                                      transform=train_transform,
                                      download=True)
    
    _train_targets = np.array(_train_dataset.targets)
    for tc in range(target_task*cls_per_task, (target_task+1)*cls_per_task):
        subset_indices += np.where(np.array(_train_dataset.targets) == tc)[0].tolist()
    

    if isinstance(replay_indices, list):
        subset_indices += replay_indices
    elif isinstance(replay_indices, np.ndarray):
        subset_indices += replay_indices.tolist()
    else:
        assert False
    
    ut, uc = np.unique(_train_targets[subset_indices], return_counts=True)
    print(ut)
    print(uc)


    train_loader = torch.utils.data.DataLoader(
        _train_dataset, batch_size=500, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=True
    )

    return train_loader
    



def set_loader_cifar10_ncm_val(cfg, normalize):

    # 値の定義
    size = cfg.dataset.size
    target_task = cfg.linear.target_task
    cls_per_task = cfg.continual.cls_per_task
    batch_size = cfg.linear.train.batch_size
    num_workers = cfg.num_workers

    val_transform = transforms.Compose([
        transforms.Resize(size=(size, size)),
        transforms.ToTensor(),
        normalize,
    ])

    target_classes = list(range(0, (target_task+1)*cls_per_task))

    subset_indices = []
    _val_dataset = datasets.CIFAR10(root=cfg.dataset.folder,
                                    train=False,
                                    transform=val_transform)
    for tc in target_classes:
        subset_indices += np.where(np.array(_val_dataset.targets) == tc)[0].tolist()
    val_dataset =  Subset(_val_dataset, subset_indices)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=500, shuffle=False,
        num_workers=8, pin_memory=True)

    return val_loader



