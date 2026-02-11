


import torch
import torch.backends.cudnn as cudnn


def make_model(cfg):

    # 使用する backbone
    backbone = cfg.model.backbone

    # head 部分の設定
    head = cfg.model.head
    feat_dim = cfg.model.feat_dim

    # 学習用データセット
    dataset = cfg.dataset.name

    if dataset in ["cifar10", "cifar100", "tiny-imagenet"]:
        from models.resnet_cifar import SupConResNet

        if backbone in ["resnet20", "resnet32", "resnet44", "resnet56"]:
            from models.resnet_cifar import SupConResNet
        
        elif backbone in ["resnet18", "resnet50"]:
            from models.resnet_cifar_co2l import SupConResNet

        model = SupConResNet(name=backbone, head=head, feat_dim=feat_dim, cfg=cfg)

    elif dataset in ["imagenet100"]:
        assert False
    

    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            model.encoder = torch.nn.DataParallel(model.encoder)
        model = model.cuda()
        cudnn.benchmark = True


    print(model)

    return model


