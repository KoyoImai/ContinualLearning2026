


import math


class BaseLearner(object):
    def __init__(self, cfg, model, model2, model_temp, criterion, optimizer, replay_indices, writer):

        # 初期値
        self.cfg = cfg
        self.model = model
        self.model2 = model2
        self.model_temp = model_temp
        self.criterion = criterion
        self.optimizer = optimizer
        self.replay_indices = replay_indices
        
        self.writer = writer
        self.global_step = 0

    # 訓練処理
    def train(self):
        pass

    # 後処理
    def post_process(self):
        pass
    
    # fc層の拡張
    def set_fc(self):
        pass
    
    # ncm 分類
    def ncm_eval(self):
        pass

    # tensorboardで可視化するための特徴
    def embedding(self, train_loader, replay_loader):
        pass