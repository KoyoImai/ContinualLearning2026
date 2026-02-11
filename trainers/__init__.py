

from trainers.supcon import SupConTrainer


def setup_trainer(cfg, model, model2, criterion, optimizer):

    if cfg.criterion.name == "ce":
        assert False
    elif cfg.criterion.name == "supcon":
        trainer = SupConTrainer(cfg, model, model2, criterion, optimizer)
    else:
        assert False

    return trainer



