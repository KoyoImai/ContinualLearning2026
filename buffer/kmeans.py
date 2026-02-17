
import math
import random
import numpy as np

import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, Subset
import torch.nn.functional as F



@torch.no_grad()
def _kmeans_select_representatives(X: torch.Tensor, k: int, iters: int = 20, seed: int = 777):
    """
    X: 特徴量(N, D)
    k: クラスタリングしたいクラスタ数
    return: selected row indices (0..N-1) length k
    """
    N = X.size(0)

    if k <= 0:
        return []
    if k >= N:
        return list(range(N))

    # deterministic
    g = torch.Generator(device=X.device)
    g.manual_seed(int(seed))

    # centers をランダムに初期化
    init = torch.randperm(N, generator=g, device=X.device)[:k]
    C = X[init].clone()  # (k, D)

    # 
    for _ in range(iters):
        
        # サンプル毎に中心との距離を計算
        # dist: (N, k)
        dist = torch.cdist(X, C)  # Euclidean
        
        # 最も近い中心を新しいクラスタとして割り当て
        labels = dist.argmin(dim=1)  # (N,)

        # 新しいクラスタの中心点を初期化
        C_new = torch.zeros_like(C)

        # 各クラスタにサンプルがいくつ割り当てられたかを計算
        counts = torch.bincount(labels, minlength=k).to(X.dtype).to(X.device)

        # 新しいクラスタの中心を計算
        C_new.index_add_(0, labels, X)
        empty = counts == 0
        counts[empty] = 1.0
        C_new = C_new / counts.view(-1, 1)

        # 空のクラスタを処理
        # handle empty cluster: re-seed with farthest points
        if empty.any():
            # closest distance to any center
            closest = dist.min(dim=1).values
            num_empty = int(empty.sum().item())
            far_ids = torch.topk(closest, k=num_empty, largest=True).indices
            C_new[empty] = X[far_ids]

        C = C_new

    # 各クラスタから，中心に近いサンプルを一つずつ選択して取り出す
    dist = torch.cdist(X, C)
    labels = dist.argmin(dim=1)

    selected = []
    used = set()

    # k 個のクラスタを一つずつ処理
    for j in range(k):

        # j 番目のクラスタのみ取り出すためのマスク
        mask = labels == j
        if not torch.any(mask):
            continue


        # j 番目のクラスタに含まれるサンプル と j番目のクラスタの中心点 との距離を計算
        d = dist[mask, j]

        # 最も中心に近い点を取り出す
        local = torch.argmin(d).item()

        # 元々の 特徴量X における行番号に直す
        global_pos = torch.nonzero(mask, as_tuple=False).view(-1)[local].item()
        if global_pos not in used:
            selected.append(global_pos)
            used.add(global_pos)

    if len(selected) < k:
        rest = [i for i in range(N) if i not in used]
        random.Random(int(seed)).shuffle(rest)
        selected += rest[: (k - len(selected))]

    return selected





@torch.no_grad()
def _extract_embeddings(cfg, model, dataset, indices, batch_size: int = 256, use_norm: bool = True):
    """
    クラス c の indices に対応する embedding (N,D) を CPU tensor で返す
    """
    m = model.module if hasattr(model, "module") else model
    device = next(m.parameters()).device

    # クラス c のみのデータローダーを作成
    loader = DataLoader(
        Subset(dataset, indices),
        batch_size=batch_size,
        shuffle=False,
        num_workers=getattr(cfg, "num_workers", 0),
        pin_memory=True,
    )

    # クラス c の特徴量を取り出す
    feats = []
    for images, _ in loader:
        images = images.to(device, non_blocking=True)
        encoded = m.encoder(images)  # universal (SupConResNet / SupCEResNet)
        if use_norm:
            encoded = F.normalize(encoded, dim=1)
        feats.append(encoded.detach().cpu())

    return torch.cat(feats, dim=0)


def set_replay_sample_kmeans(cfg, model, prev_indices):

    # 長い設定変数を先に取り出しておく
    target_task = cfg.continual.target_task
    cls_per_task = cfg.continual.cls_per_task
    mem_size = cfg.buffer.size


    is_training = model.training
    model.eval()

    # データローダの仮作成（ラベルがほしいだけ）
    val_transform = transforms.Compose([
        transforms.Resize(cfg.dataset.size),
        transforms.ToTensor(),
    ])

    if cfg.dataset.name == "cifar10":
        subset_indices = []
        val_dataset = datasets.CIFAR10(root="/home/kouyou/datasets/",
                                       transform=val_transform,
                                       download=True)
        val_targets = np.array(val_dataset.targets)

    elif cfg.dataset.name == 'cifar100':
        subset_indices = []
        val_dataset = datasets.CIFAR100(root="/home/kouyou/datasets/",
                                        transform=val_transform,
                                        download=True)
        val_targets = np.array(val_dataset.targets)

    elif cfg.dataset.name == 'tiny-imagenet':
        subset_indices = []
        val_dataset = TinyImagenet(root="/home/kouyou/datasets/",
                                   transform=val_transform,
                                   download=True)
        val_targets = val_dataset.targets
    
    else:
        raise ValueError('dataset not supported: {}'.format(cfg.dataset.name))


    # すでにバッファ内のサンプルに対して，バッファに保持するデータを選択
    if prev_indices is None:
        prev_indices = []
        observed_classes = list(range(0, target_task*cls_per_task))
    else:

        # 過去タスクのデータに割り当てるバッファのサイズ
        shrink_size = ((target_task - 1) * mem_size / target_task)

        if len(prev_indices) > 0:

            unique_cls = np.unique(val_targets[prev_indices])
            _prev_indices = prev_indices
            prev_indices = []

            # k-meansによって維持・削除するデータを決定
            for c in unique_cls:

                # クラス c の候補
                mask = val_targets[_prev_indices] == c
                cand = torch.tensor(_prev_indices)[mask].tolist()
                
                # クラス c に割り当てるバッファサイズを決定
                size_for_c = shrink_size / len(unique_cls)
                p = size_for_c - (shrink_size // len(unique_cls))
                size_for_c = math.ceil(size_for_c) if random.random() < p else math.floor(size_for_c)
                size_for_c = int(min(size_for_c, len(cand)))
                if size_for_c <= 0:
                    continue

                X = _extract_embeddings(cfg, model, val_dataset, cand, batch_size=256, use_norm=True)
                pick_local = _kmeans_select_representatives(X, k=size_for_c, iters=30, seed=cfg.seed + int(c))
                prev_indices += [cand[i] for i in pick_local]
        
        # 前回タスクのクラス範囲
        observed_classes = list(range(max(target_task - 1, 0) * cls_per_task, (target_task) * cls_per_task))

    # 確認済みのクラス（前回タスク）がない場合終了
    if len(observed_classes) == 0:
        model.train()
        return prev_indices
    
    # 確認済みクラスのインデックスを獲得
    observed_indices = []
    for tc in observed_classes:
        observed_indices += np.where(val_targets == tc)[0].tolist()

    val_observed_targets = val_targets[observed_indices]
    val_unique_cls = np.unique(val_observed_targets)
    print("val_unique_cls: ", val_unique_cls)

    selected_observed_indices = []
    for c_idx, c in enumerate(val_unique_cls):
        
        # 割り当て数に関しては random.py の“残り枠の均等割当て”をそのまま使用
        remaining = mem_size - len(prev_indices) - len(selected_observed_indices)
        remaining_cls = (len(val_unique_cls) - c_idx)
        if remaining <= 0:
            break

        size_for_c_float = remaining / remaining_cls
        p = size_for_c_float - (remaining // remaining_cls)

        if random.random() < p:
            size_for_c = math.ceil(size_for_c_float)
        else:
            size_for_c = math.floor(size_for_c_float)

        # クラス c の候補サンプルのインデックスを取り出す
        cand = torch.tensor(observed_indices)[val_targets[observed_indices] == c].tolist()
        size_for_c = int(min(size_for_c, len(cand)))
        if size_for_c <= 0:
            continue
            
        X = _extract_embeddings(cfg, model, val_dataset, cand, batch_size=256, use_norm=True)
        pick_local = _kmeans_select_representatives(X, k=size_for_c, iters=30, seed=cfg.seed + 10_000 + int(c))
        selected_observed_indices += [cand[i] for i in pick_local]

    if is_training:
        model.train()

    return prev_indices + selected_observed_indices