

import numpy as np

import torch
from torchvision import transforms, datasets
from torch.utils.data import Subset
from torch.utils.data import WeightedRandomSampler


def set_loader_cifar100_linear_train(cfg, normalize, replay_indices):

    # 値の定義
    size = cfg.dataset.size
    target_task = cfg.linear.target_task
    cls_per_task = cfg.continual.cls_per_task
    batch_size = cfg.linear.train.batch_size
    num_workers = cfg.num_workers

    train_transform = transforms.Compose([
        transforms.Resize(size=(size, size)),
        transforms.RandomResizedCrop(size=size, scale=(0.1, 1.)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=size//20*2+1, sigma=(0.1, 2.0))], p=0.5 if size>32 else 0.0),
        transforms.ToTensor(),
        normalize,
    ])

    subset_indices = []
    _train_dataset = datasets.CIFAR100(root=cfg.dataset.folder,
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

    # print("len(replay_indices): ", len(replay_indices))
    # print("len(subset_indices): ", len(subset_indices))
    # assert False

    ut, uc = np.unique(_train_targets[subset_indices], return_counts=True)
    print(ut)
    print(uc)

    weights = np.array([0.] * len(subset_indices))
    for t, c in zip(ut, uc):
        weights[_train_targets[subset_indices] == t] = 1./c

    train_dataset =  Subset(_train_dataset, subset_indices)

    train_sampler = WeightedRandomSampler(torch.Tensor(weights), len(weights))
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=(train_sampler is None),
        num_workers=cfg.num_workers, pin_memory=True, sampler=train_sampler)
    
    return train_loader



def set_loader_cifar100_linear_val(cfg, normalize):

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
    _val_dataset = datasets.CIFAR100(root=cfg.dataset.folder,
                                     train=False,
                                     transform=val_transform)
    for tc in target_classes:
        subset_indices += np.where(np.array(_val_dataset.targets) == tc)[0].tolist()
    val_dataset =  Subset(_val_dataset, subset_indices)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=256, shuffle=False,
        num_workers=8, pin_memory=True)

    return val_loader