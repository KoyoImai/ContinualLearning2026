


import math


class BaseLearner(object):
    def __init__(self, cfg, model, model2, model_temp, criterion, optimizer, writer):

        # 初期値
        self.cfg = cfg
        self.model = model
        self.model2 = model2
        self.model_temp = model_temp
        self.criterion = criterion
        self.optimizer = optimizer
        
        self.writer = writer
        self.global_step = 0

    def train(self):
        pass

    # def set_scheduler(self):

    #     if self.cfg.optimizer.scheduler.warm:

    #         # warmup_from_enc = self.cfg.optimizer.scheduler.warmup_from_enc
    #         # warmup_from_prot = self.cfg.optimizer.scheduler.warmup_from_prot
    #         # warm_epochs = self.cfg.optimizer.scheduler.warm_epochs
    #         cosine = self.cfg.optimizer.scheduler.cosine
            
    #         learning_rate = self.cfg.optimizer.learning_rate
    #         lr_decay_rate = self.cfg.optimizer.scheduler.lr_decay_rate
    #         learning_rate_prototypes = self.cfg.optimizer.learning_rate_prototypes

    #         epochs = self.cfg.train.epochs
    #         warm_epochs = self.cfg.optimizer.scheduler.warm_epochs

    #         if cosine:
    #             eta_min_encoder = learning_rate * (lr_decay_rate ** 3)
    #             eta_min_prototypes = learning_rate_prototypes * (lr_decay_rate ** 3)
    #             self.cfg.optimizer.scheduler.warmup_to_enc = eta_min_encoder + (learning_rate - eta_min_encoder) * (
    #                     1 + math.cos(math.pi * warm_epochs / epochs)) / 2
    #             self.cfg.optimizer.scheduler.warmup_to_prot = eta_min_prototypes + (learning_rate_prototypes - eta_min_prototypes) * (
    #                     1 + math.cos(math.pi * warm_epochs / epochs)) / 2
    #         else:
    #             self.cfg.optimizer.scheduler.warmup_to_enc = learning_rate
    #             self.cfg.optimizer.scheduler.warmup_to_prot = learning_rate_prototypes



