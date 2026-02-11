

from torchvision import transforms, datasets


from dataloaders.dataloader_cifar10 import set_loader_cifar10

from dataloaders.linear_cifar10 import set_loader_cifar10_linear_train, set_loader_cifar10_linear_val


def set_loader(cfg, model, replay_indices):

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
        train_loader, subset_indices = set_loader_cifar10(cfg, normalize, replay_indices)

    else:
        assert False


    return train_loader, subset_indices


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