

import torch


def make_criterion(cfg):

    name = cfg.criterion.name

    if name == "ce":
        assert False
    elif name == "supcon":
        from criterions.supcon import SupConLoss
        criterion = SupConLoss(temperature=cfg.criterion.temp)
    elif name == "asym_supcon":
        from criterions.asym_supcon import AsymSupConLoss
        criterion = AsymSupConLoss(temperature=cfg.criterion.temp)
    elif name == "proto_supcon":
        from criterions.proto_supcon import ProtoSupConLoss
        criterion = ProtoSupConLoss(temperature=cfg.criterion.temp, cfg=cfg)
    elif name == "is_supcon":
        from criterions.cclis import ISSupConLoss
        criterion = ISSupConLoss(temperature=cfg.criterion.temp, cfg=cfg)
    else:
        assert False


    if torch.cuda.is_available():
        criterion = criterion.cuda()


    return criterion





