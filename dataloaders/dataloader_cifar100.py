import numpy as np

import torch
from torchvision import transforms,datasets
from torch.utils.data import Subset
from torch.utils.data.dataset import ConcatDataset
from torch.utils.data import Sampler, RandomSampler


class TwoCropTransform:
    """Create two crops of the same image"""
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        return [self.transform(x), self.transform(x)]
    

#=============================
# Samplerの作成
#=============================
class BatchSchedulerSampler(torch.utils.data.sampler.Sampler):

    def __init__(self, dataset, batch_size):

        self.dataset = dataset
        self.batch_size = batch_size
        self.number_of_datasets = len(dataset.datasets) 

        self.largest_dataset_size = max([len(cur_dataset) for cur_dataset in dataset.datasets])
        self.dataset_len = sum([len(cur_dataset) for cur_dataset in self.dataset.datasets])

    def __len__(self):
        return self.dataset_len
    
    def __iter__(self):

        samplers_list = []
        sampler_iterators = []

        for dataset_idx in range(self.number_of_datasets):

            # データセットを 1 つ取り出す
            cur_dataset = self.dataset.datasets[dataset_idx]

            # samplerの作成
            sampler = RandomSampler(cur_dataset) 
            samplers_list.append(sampler)

            cur_sampler_iterator = sampler.__iter__()
            sampler_iterators.append(cur_sampler_iterator)

        # 各サブデータセットの開始位置を獲得
        push_index_val = [0] + self.dataset.cumulative_sizes[:-1] 

        # ミニバッチサイズの合計
        step = sum(self.batch_size) 

        samples_to_grab = self.batch_size 
        epoch_samples = self.dataset_len  


        # 最終的に返す index 列の初期化
        final_samples_list = []

        for _ in range(0, epoch_samples, step):
            for i in range(self.number_of_datasets):
                cur_batch_sampler = sampler_iterators[i] 
                cur_samples = []

                # リプレイバッファ内のサンプルと現在タスクのサンプルを分けて取り出す
                for _ in range(samples_to_grab[i]):

                    try:
                        cur_sample_org = cur_batch_sampler.__next__()
                        cur_sample = cur_sample_org + push_index_val[i]
                        cur_samples.append(cur_sample)
                    
                    except StopIteration: 
                        # got to the end of iterator - restart the iterator and continue to get samples
                        # until reaching "epoch_samples"
                        break
                
                final_samples_list.extend(cur_samples)


        return iter(final_samples_list)
    



#=============================
# CIFAR100 データローダー
#=============================
def set_loader_cifar100(cfg, normalize, replay_indices, training=True):


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
    _train_dataset = datasets.CIFAR100(root=cfg.dataset.folder,
                                       transform=TwoCropTransform(train_transform),
                                       download=True)

    for tc in target_classes:
        target_class_indices = np.where(np.array(_train_dataset.targets) == tc)[0]
        subset_indices += np.where(np.array(_train_dataset.targets) == tc)[0].tolist()


    if len(replay_indices) > 0 and training:

        # 現在タスクのサンプルでデータセットを作成
        cur_dataset = Subset(_train_dataset, subset_indices)

        # リプレイバッファ内のサンプルでデータセットを作成
        prev_dataset = Subset(_train_dataset, replay_indices)

        # 各データセットのサイズを保存
        dataset_len_list = [len(prev_dataset), len(cur_dataset)]

        # 現在タスクとリプレイバッファ内のサンプルのデータセットを concat
        train_dataset = ConcatDataset([prev_dataset, cur_dataset])

    else:
        
        # 現在タスクのデータセットのみでデータセットを作成
        train_dataset = Subset(_train_dataset, subset_indices)
    

    # 現在タスクのサンプルのインデックスとリプレイバッファ内のサンプルのインデックスを合算
    _subset_indices = subset_indices + replay_indices

    # 訓練用データセット内の内訳を表示
    uk, uc = np.unique(np.array(_train_dataset.targets)[_subset_indices], return_counts=True)  
    print('uc[np.argsort(uk)]', uc[np.argsort(uk)])
    replay_sample_num = uc[np.argsort(uk)]


    if len(replay_indices) > 0 and training:

        # ミニバッチ内の「現在タスクのサンプル」と「バッファ内のサンプル」の比率を決定
        train_batch_size_list = [int(np.round(batch_size * dataset_len_list[0] / sum(dataset_len_list))), batch_size - int(np.round(batch_size * dataset_len_list[0] / sum(dataset_len_list)))]

        print('train_batch_size', train_batch_size_list)
        train_sampler = BatchSchedulerSampler(dataset=train_dataset, batch_size=train_batch_size_list)

    else:
        train_sampler = None
    

    if training:
        train_loader = torch.utils.data.DataLoader(
                            train_dataset, batch_size=batch_size, shuffle=(train_sampler is None),
                            num_workers=num_workers, pin_memory=True, sampler=train_sampler)


    else:
        train_loader = torch.utils.data.DataLoader(
                            train_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)
        print('no separate sampler')


    # # Samplerが正しく機能しているかの確認
    # for b, (imgs, labels) in enumerate(train_loader):

    #     # 2) クラス別枚数
    #     uniq, cnt = labels.unique(return_counts=True)
    #     print("  class_counts:", dict(zip(uniq.tolist(), cnt.tolist())))
    

    return train_loader, _subset_indices


def set_vanila_loader_cifar100(cfg, normalize, replay_indices):

    # 値の定義
    size = cfg.dataset.size
    target_task = cfg.continual.target_task
    cls_per_task = cfg.continual.cls_per_task
    batch_size = cfg.train.batch_size
    num_workers = cfg.num_workers

    # データ拡張の定義
    train_transform = transforms.Compose([
        transforms.Resize(size=(size, size)),
        transforms.ToTensor(),
        normalize,
    ])

    # 現在タスクまでの全クラス
    target_classes = list(range(0, (target_task+1)*cls_per_task))
    print(target_classes)

    # 現在タスクのクラスのみを対象にインデックスを取り出す
    subset_indices = []
    _train_dataset = datasets.CIFAR100(root=cfg.dataset.folder,
                                       transform=train_transform,
                                       download=True)

    for tc in target_classes:
        target_class_indices = np.where(np.array(_train_dataset.targets) == tc)[0]
        subset_indices += np.where(np.array(_train_dataset.targets) == tc)[0].tolist()


    # 現在タスクのサンプルのインデックスとリプレイバッファ内のサンプルのインデックスを合算しデータセットを作成
    # subset_indices += replay_indices
    train_dataset =  Subset(_train_dataset, subset_indices)
    replay_dataset = Subset(_train_dataset, replay_indices)

    # 訓練用データセット内の内訳を表示
    uk, uc = np.unique(np.array(_train_dataset.targets)[subset_indices], return_counts=True)  
    print('uc[np.argsort(uk)]', uc[np.argsort(uk)])

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=500, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )

    replay_laoder = torch.utils.data.DataLoader(
        replay_dataset, batch_size=500, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )

    vanila_loaders = {"train": train_loader, "replay": replay_laoder}

    return vanila_loaders

