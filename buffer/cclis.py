
import math
import random
import numpy as np

import torch
from torchvision import transforms, datasets
from torch.utils.data import Dataset, Subset



def set_replay_samples_cclis(cfg, model, trainer, prev_indices):

    # cclis のデータ選択に必要な値
    prev_importance_weight = trainer.importance_weight
    prev_score = trainer.score

    target_task = cfg.continual.target_task
    cls_per_task = cfg.continual.cls_per_task
    mem_size = cfg.buffer.size

    class IdxDataset(Dataset):
        def __init__(self, dataset, indices):
            self.dataset = dataset
            self.indices = indices
        def __len__(self):
            return len(self.dataset)
        def __getitem__(self, idx):
            return self.indices[idx], self.dataset[idx]

    # スコア計算用のデータ拡張
    val_transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    if cfg.dataset.name == "cifar10":
        subset_indices = []
        val_dataset = datasets.CIFAR10(root=cfg.dataset.folder,
                                       transform=val_transform,
                                       download=True)
        val_targets = np.array(val_dataset.targets)
    elif cfg.dataset.name == "cifar100":
        subset_indices = []
        val_dataset = datasets.CIFAR100(root=cfg.dataset.folder,
                                        transform=val_transform,
                                        download=True)
        val_targets = np.array(val_dataset.targets)
    else:
        assert False

    prev_indices_len = 0

    # リプレイバッファにデータが存在しない場合（最初のタスクを学習する直前）
    if prev_indices is None:

        # バッファ内データのインデックスと importane_weight の初期化
        prev_indices, prev_importance_weight = [], []
        observed_classes = list(range(0, target_task*cls_per_task))

    # リプレイバッファ内にデータが存在する場合
    else:

        # 縮小サイズ（過去タスクのデータに割り当てれるメモリ）
        shrink_size = ((target_task - 1) * mem_size / target_task)

        if len(prev_indices):

            unique_cls = np.unique(val_targets[prev_indices])
            _prev_indices = prev_indices
            prev_indices_len = len(prev_indices)
            prev_indices = []
            prev_weight = prev_importance_weight 
            prev_importance_weight = []

            # 1クラスずつ処理
            for c in unique_cls:
                mask = val_targets[_prev_indices] == c

                size_for_c = shrink_size / len(unique_cls)
                p = size_for_c - (shrink_size // len(unique_cls))  

                if random.random() < p:
                    size_for_c = math.ceil(size_for_c)
                else:
                    size_for_c = math.floor(size_for_c)

                store_index = torch.multinomial(torch.tensor(prev_score[:prev_indices_len])[mask], min(len(torch.tensor(prev_score[:prev_indices_len])[mask]), size_for_c), replacement=False)  # score tensor [old_samples_num] 

                prev_indices += torch.tensor(_prev_indices)[mask][store_index].tolist()

                prev_cur_weight = torch.tensor(prev_score[:prev_indices_len])[mask]

                prev_importance_weight += (prev_cur_weight / prev_cur_weight.sum())[store_index].tolist()

            print(np.unique(val_targets[prev_indices], return_counts=True))
        observed_classes = list(range(max(target_task-1, 0)*cls_per_task, (target_task)*cls_per_task))

    # 学習済みクラスが存在しない場合
    if len(observed_classes) == 0:
        return prev_indices, prev_importance_weight, val_targets

    # 観測直後のタスクのクラス（1タスク前のクラス）
    observed_indices = []
    for tc in observed_classes:
        observed_indices += np.where(val_targets == tc)[0].tolist()

    # ラベルの獲得
    val_observed_targets = val_targets[observed_indices]
    val_unique_cls = np.unique(val_observed_targets)

    # 新しくバッファに入れるサンプルのインデックスと importance weight を初期化
    selected_observed_indices = []
    selected_observed_importance_weight = []

    # 1クラスずつ順番に処理を実行
    for c_idx, c in enumerate(val_unique_cls):
        
        # クラス c に割り当てるバッファサイズの決定
        size_for_c_float = ((mem_size - len(prev_indices) - len(selected_observed_indices)) / (len(val_unique_cls) - c_idx))
        p = size_for_c_float -  ((mem_size - len(prev_indices) - len(selected_observed_indices)) // (len(val_unique_cls) - c_idx))
        if random.random() < p:
            size_for_c = math.ceil(size_for_c_float)
        else:
            size_for_c = math.floor(size_for_c_float)

        # 特定クラスcのみを取り出すためのマスク
        mask = val_targets[observed_indices] == c

        # この下2行をコメントアウトしたら，さらに下の「store_index=〜」のコメントアウトを変更
        scores = torch.tensor(prev_score[prev_indices_len:])[mask]
        size_for_c = min(size_for_c, len(scores))  # エラー防止
        
        # prev_scoreをもとにサンプル毎に重みづけをして保存するサンプルを選択
        # store_index = torch.multinomial(torch.tensor(prev_score[prev_indices_len:])[mask], size_for_c, replacement=False)
        store_index = torch.multinomial(scores, size_for_c, replacement=False)

    # 選択されたサンプルのインデックスを蓄積
        selected_observed_indices += torch.tensor(observed_indices)[mask][store_index].tolist()

        # 特定クラスcに属する全サンプルのスコアを取り出す（提案分布）
        observed_cur_weight = torch.tensor(prev_score[prev_indices_len:])[mask] 
        
        # スコアを正規化
        observed_normalized_weight = observed_cur_weight / observed_cur_weight.sum() 

        # 保存するサンプルのスコアのみを取り出す
        selected_observed_importance_weight += observed_normalized_weight[store_index].tolist()  

    print(np.unique(val_targets[selected_observed_indices], return_counts=True))
    print(selected_observed_importance_weight)

    return prev_indices + selected_observed_indices, prev_importance_weight + selected_observed_importance_weight, val_targets

