


import os
import copy
import torch
import hydra
import logging
import numpy as np

from utils import seed_setup, save_model
from models import make_model                  # mode の作成
from trainers import setup_trainer             # trainer の作成
from dataloaders import set_loader_linear      # dataloader の作成



@hydra.main(config_path='configs/default/', config_name='default', version_base=None)
def main(cfg):

    #=======================
    # seed 値の固定
    #=======================
    seed_setup(cfg.seed)

    #=======================
    # model の作成
    #=======================
    model = make_model(cfg=cfg)

    # パラメータの読み込み
    # ckpt_path = f"{cfg.log.model_path}/model_{cfg.linear.target_task:02d}.pth"
    ckpt_path = f"{cfg.log.model_path}/model_init.pth"
    ckpt = torch.load(ckpt_path, map_location='cpu')
    state_dict = ckpt['model']
    model.load_state_dict(state_dict)

    #=======================
    # データローダーの作成
    #=======================
    # リプレイサンプルの読み込み
    if cfg.linear.target_task == 0:
        replay_indices = np.array([])
    else:
        # file_path = f"{cfg.log.mem_path}/replay_indices_{cfg.linear.target_task}.npy"
        file_path = f"/home/kouyou/ContinualLearning2026/debug/replay_indices_4.npy"
        # file_path = f"{opt.log_path}/replay_indices_0.npy"
        replay_indices = np.load(file_path)

    # データローダーの作成（バッファ内のデータも含めて）
    train_loader, val_loader = set_loader_linear(cfg, model, replay_indices)

    #=======================
    # trainer の作成
    #=======================
    # trainer = setup_trainer(cfg, model, model2, criterion, optimizer)
    trainer = setup_trainer(cfg, model, None, None, None)

    trainer.linear_eval(train_loader, val_loader)



if __name__ == "__main__":
    main()


