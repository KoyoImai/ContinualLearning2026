

from torchvision import transforms, datasets

# 通常のデータローダーの作成
from dataloaders.dataloader_cifar10 import set_loader_cifar10, set_vanila_loader_cifar10
from dataloaders.dataloader_cifar100 import set_loader_cifar100, set_vanila_loader_cifar100

# cclis用データローダーの作成
from dataloaders.dataloader_cclis_cifar10 import set_loader_cclis_cifar10
from dataloaders.dataloader_cclis_cifar100 import set_loader_cclis_cifar100

# 線形分類による評価用データローダーの作成
from dataloaders.linear_cifar10 import set_loader_cifar10_linear_train, set_loader_cifar10_linear_val

# NCM分類による評価用データローダーの作成
from dataloaders.ncm_cifar10 import set_loader_cifar10_ncm_train, set_loader_cifar10_ncm_val

def set_loader(cfg, trainer, replay_indices):

    """
    return
        train_loader   : 訓練に使用するデータローダー
        vanila_loaders : 訓練後に特徴埋め込みを獲得するためのデータローダー
    """

    # 正規化パラメータの設定
    if cfg.dataset.name == "cifar10":
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
    elif cfg.dataset.name == "cifar100":
        mean = (0.5071, 0.4867, 0.4408)
        std = (0.2675, 0.2565, 0.2761)
    elif cfg.dataset.name == "tiny-imagenet":
        mean = (0.4802, 0.4480, 0.3975)
        std = (0.2770, 0.2691, 0.2821)
    elif cfg.dataset.name == "imagenet100":
        assert False
    else:
        assert False

    normalize = transforms.Normalize(mean=mean, std=std)


    # データローダーの作成
    if cfg.dataset.name == "cifar10":
        if cfg.criterion.name == "is_supcon":
            train_loader, subset_indices, subset_sample_num = set_loader_cclis_cifar10(cfg, normalize, trainer, replay_indices)
            post_loader, _, _ = set_loader_cclis_cifar10(cfg, normalize, trainer, replay_indices, training=False)
            vanila_loaders = set_vanila_loader_cifar10(cfg, normalize, replay_indices)

            trainer.subset_sample_num = subset_sample_num
            trainer.post_loader = post_loader
        else:
            train_loader, subset_indices = set_loader_cifar10(cfg, normalize, replay_indices)
            vanila_loaders = set_vanila_loader_cifar10(cfg, normalize, replay_indices)
    
    elif cfg.dataset.name == "cifar100":
        if cfg.criterion.name == "is_supcon":

            train_loader, subset_indices, subset_sample_num = set_loader_cclis_cifar100(cfg, normalize, trainer, replay_indices)
            post_loader, _, _ = set_loader_cclis_cifar100(cfg, normalize, trainer, replay_indices, training=False)
            vanila_loaders = set_vanila_loader_cifar100(cfg, normalize, replay_indices)

            trainer.subset_sample_num = subset_sample_num
            trainer.post_loader = post_loader
        
        else:
            train_loader, subset_indices = set_loader_cifar100(cfg, normalize, replay_indices)
            vanila_loaders = set_vanila_loader_cifar100(cfg, normalize, replay_indices)

    else:
        assert False


    return train_loader, vanila_loaders, subset_indices


def set_loader_linear(cfg, model, replay_indices):

    # 正規化パラメータの設定
    if cfg.dataset.name == "cifar10":
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
    elif cfg.dataset.name == "cifar100":
        mean = (0.5071, 0.4867, 0.4408)
        std = (0.2675, 0.2565, 0.2761)
    elif cfg.dataset.name == "tiny-imagenet":
        mean = (0.4802, 0.4480, 0.3975)
        std = (0.2770, 0.2691, 0.2821)
    elif cfg.dataset.name == "imagenet100":
        assert False
    else:
        assert False

    normalize = transforms.Normalize(mean=mean, std=std)

    if cfg.dataset.name == "cifar10":

        train_loader = set_loader_cifar10_linear_train(cfg, normalize, replay_indices)
        val_loader = set_loader_cifar10_linear_val(cfg, normalize)

    else:
        assert False

    
    return train_loader, val_loader




def set_loader_ncm(cfg, model, replay_indices):

    # 正規化パラメータの設定
    if cfg.dataset.name == "cifar10":
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
    elif cfg.dataset.name == "cifar100":
        mean = (0.5071, 0.4867, 0.4408)
        std = (0.2675, 0.2565, 0.2761)
    elif cfg.dataset.name == "tiny-imagenet":
        mean = (0.4802, 0.4480, 0.3975)
        std = (0.2770, 0.2691, 0.2821)
    elif cfg.dataset.name == "imagenet100":
        assert False
    else:
        assert False

    normalize = transforms.Normalize(mean=mean, std=std)

    if cfg.dataset.name == "cifar10":

        train_loader = set_loader_cifar10_ncm_train(cfg, normalize, replay_indices)
        val_loader = set_loader_cifar10_ncm_val(cfg, normalize)

    else:
        assert False

    
    return train_loader, val_loader