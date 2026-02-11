

from buffer.random import set_replay_samples_random


def set_buffer(cfg, model, prev_indices=None):

    type = cfg.buffer.type
    size = cfg.buffer.size

    if type == "random":
        replay_indices = set_replay_samples_random(cfg, model, prev_indices=prev_indices)
    elif type == "aaa":
        assert False
    else:
        assert False
    
    # print("replay_indices: ", replay_indices)
    # print("len(replay_indices): ", len(replay_indices))

    return replay_indices





