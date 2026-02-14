

from buffer.random import set_replay_samples_random
from buffer.cclis import set_replay_samples_cclis


def set_buffer(cfg, model, trainer, prev_indices=None):

    type = cfg.buffer.type
    size = cfg.buffer.size

    if type == "random":
        replay_indices = set_replay_samples_random(cfg, model, prev_indices=prev_indices)
    elif type == "cclis":
        replay_indices, importance_weight, val_targets = set_replay_samples_cclis(cfg, model, trainer, prev_indices=prev_indices)
        trainer.importance_weight = importance_weight
        trainer.val_targets = val_targets
    else:
        assert False
    
    # print("replay_indices: ", replay_indices)
    # print("len(replay_indices): ", len(replay_indices))

    return replay_indices





