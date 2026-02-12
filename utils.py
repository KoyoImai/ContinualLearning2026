

import math
import random
import torch
import numpy as np




#=====================
# seed値の固定
#=====================
def seed_setup(seed):

    # Python 標準の乱数生成器のシード固定
    random.seed(seed)
    
    # NumPy の乱数生成器のシード固定
    np.random.seed(seed)
    
    # PyTorch のシード固定
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        
    # Deterministic モードの有効化（PyTorch の一部非決定的な処理の回避）
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


#=====================
# モデルを保存
#=====================
def save_model(model, optimizer, cfg, epoch, save_file):
    print('==> Saving...'+save_file)
    state = {
        'cfg': cfg,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
    }
    torch.save(state, save_file)
    del state


# #=====================
# # Schedulerの処理
# #=====================
# def adjust_learning_rate(cfg, optimizer, epoch):

#     learning_rate = cfg.optimizer.learning_rate
#     learning_rate_prototypes = cfg.optimizer.learning_rate_prototypes
#     cosine = cfg.optimizer.scheduler.cosine
#     lr_decay_rate = cfg.optimizer.scheduler.lr_decay_rate

#     lr_enc = learning_rate
#     lr_prot = learning_rate_prototypes
    
#     if cosine:
#         eta_min_enc = lr_enc * (lr_decay_rate ** 3)
#         eta_min_prot = lr_prot * (lr_decay_rate ** 3)
#         lr_enc = eta_min_enc + (lr_enc - eta_min_enc) * (
#                 1 + math.cos(math.pi * epoch / cfg.train.epochs)) / 2
#         lr_prot = eta_min_prot + (lr_prot - eta_min_prot) * (
#                 1 + math.cos(math.pi * epoch / cfg.train.epochs)) / 2        
#     else:
#         steps = np.sum(epoch > np.asarray(cfg.optimizer.scheduler.lr_decay_epochs))
#         if steps > 0:
#             lr_enc = lr_enc * (lr_decay_rate ** steps)
#             lr_prot = lr_prot * (lr_decay_rate ** steps)

#     lr_list = [lr_enc, lr_enc, lr_prot]

#     for idx, param_group in enumerate(optimizer.param_groups):
#         # print('idx: ', idx)
#         param_group['lr'] = lr_list[idx]


# def warmup_learning_rate(cfg, epoch, batch_id, total_batches, optimizer):
    
#     warm = cfg.optimizer.scheduler.warm
#     warm_epochs = cfg.optimizer.scheduler.warm_epochs
#     warmup_from_enc = cfg.optimizer.scheduler.warmup_from_enc
#     warmup_from_prot = cfg.optimizer.scheduler.warmup_from_prot
#     warmup_to_enc = cfg.optimizer.scheduler.warmup_to_enc
#     warmup_to_prot = cfg.optimizer.scheduler.warmup_to_prot

#     if warm and epoch <= warm_epochs:
#         p = (batch_id + (epoch - 1) * total_batches) / \
#             (warm_epochs * total_batches)
#         lr_enc = warmup_from_enc + p * (warmup_to_enc - warmup_from_enc)
#         lr_prot = warmup_from_prot + p * (warmup_to_prot - warmup_from_prot)
#         lr_list = [lr_enc, lr_enc, lr_prot]

#         for idx, param_group in enumerate(optimizer.param_groups):
#             # print("lr_list[idx]: ", lr_list[idx])

#             # print('idx: ', idx)
#             param_group['lr'] = lr_list[idx]

#=====================
# 学習記録
#=====================
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count




