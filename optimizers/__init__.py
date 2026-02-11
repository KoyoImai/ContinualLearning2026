


import torch.optim as optim


def make_optimizer(cfg, model):

    # 最適化手法
    name = cfg.optimizer.name 

    # ハイパラ
    learning_rate=cfg.optimizer.learning_rate
    momentum=cfg.optimizer.momentum
    weight_decay=cfg.optimizer.weight_decay
    learning_rate_prototypes = cfg.optimizer.learning_rate_prototypes

    if name == "sgd":
        if 'prototypes.weight' in model.state_dict().keys():
            optimizer = optim.SGD([
                            {'params': model.encoder.parameters()},
                            {'params': model.head.parameters()},
                            {'params': model.prototypes.parameters(), 'lr': learning_rate_prototypes},
                            ],
                            lr=learning_rate,
                            momentum=momentum,
                            weight_decay=weight_decay)
        else:
            optimizer = optim.SGD(model.parameters(),
                            lr=learning_rate,
                            momentum=momentum,
                            weight_decay=weight_decay)
    else:
        assert False


    return optimizer





