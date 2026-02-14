

from trainers.ce import CETrainer
from trainers.supcon import SupConTrainer
from trainers.asym_supcon import AsymSupConTrainer
from trainers.proto_supcon import ProtoSupConTrainer
from trainers.cclis import CCLISTrainer


def setup_trainer(cfg, model, model2, model_temp, criterion, optimizer, replay_indices, writer):

    if cfg.criterion.name == "ce":
        trainer = CETrainer(cfg, model, model2, model_temp, criterion, optimizer, replay_indices, writer)
    elif cfg.criterion.name in ["supcon"]:
        trainer = SupConTrainer(cfg, model, model2, model_temp, criterion, optimizer, replay_indices, writer)
    elif cfg.criterion.name in ["asym_supcon"]:
        trainer = AsymSupConTrainer(cfg, model, model2, model_temp, criterion, optimizer, replay_indices, writer)
    elif cfg.criterion.name in ["proto_supcon"]:
        trainer = ProtoSupConTrainer(cfg, model, model2, model_temp, criterion, optimizer, replay_indices, writer)
    elif cfg.criterion.name in ["is_supcon"]:
        trainer = CCLISTrainer(cfg, model, model2, model_temp, criterion, optimizer, replay_indices, writer)
    else:
        assert False

    return trainer



