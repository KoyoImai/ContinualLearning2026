

from trainers.supcon import SupConTrainer
from trainers.asym_supcon import AsymSupConTrainer
from trainers.proto_supcon import ProtoSupConTrainer


def setup_trainer(cfg, model, model2, model_temp, criterion, optimizer, writer):

    if cfg.criterion.name == "ce":
        assert False
    elif cfg.criterion.name in ["supcon"]:
        trainer = SupConTrainer(cfg, model, model2, model_temp, criterion, optimizer, writer)
    elif cfg.criterion.name in ["asym_supcon"]:
        trainer = AsymSupConTrainer(cfg, model, model2, model_temp, criterion, optimizer, writer)
    elif cfg.criterion.name in ["proto_supcon"]:
        trainer = ProtoSupConTrainer(cfg, model, model2, model_temp, criterion, optimizer, writer)
    else:
        assert False

    return trainer



