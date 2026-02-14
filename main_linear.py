


import os
import copy
import torch
import hydra
import logging
import numpy as np

from utils import seed_setup, save_model, make_dir
from models import make_model                  # mode の作成
from trainers import setup_trainer             # trainer の作成
from dataloaders import set_loader_linear      # dataloader の作成


def setup_logging(cfg):
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        # handlers=[
        #     logging.StreamHandler(),                   # コンソール出力
        #     logging.FileHandler(f"{cfg.hydra.run.dir}/experiment.log", mode="w")  # ファイルに記録（上書きモード）
        # ]
    )


@hydra.main(config_path='configs/default/', config_name='default', version_base=None)
def main(cfg):

    #=======================
    # seed 値の固定
    #=======================
    seed_setup(cfg.seed)


    #=======================
    # log の名前を設定（未実装）
    #=======================
    log_base = (f"logs/{cfg.log.name}_{cfg.seed}_{cfg.date}/"                                           # logの名前（分析やデバッグなど，目的に沿って名前をconfigファイルで指定）
                f"{cfg.dataset.name}/"                                                                  # データセット名 で ディレクトリ分岐
                f"{cfg.continual.n_task}_{cfg.continual.cls_per_task}/"                                 # タスク設定 で ディレクトリ分岐
                f"{cfg.model.backbone}_{cfg.model.head}_"                                               # モデルやOptimizerなど で ディレクトリ分岐
                f"{cfg.criterion.name}_{cfg.criterion.distill.type}{cfg.criterion.distill.power}_"
                f"{cfg.optimizer.name}_"
                f"{cfg.buffer.type}{cfg.buffer.size}")
    cfg.log.base = log_base

    # loggerの設定
    setup_logging(cfg=cfg)
    logging.info("Experiment started")

    # 実験記録を保存するディレクトリを作成
    make_dir(cfg)


    #=======================
    # model の作成
    #=======================
    model = make_model(cfg=cfg)

    # パラメータの読み込み
    ckpt_path = f"{cfg.log.model_path}/model_{cfg.linear.target_task:02d}.pth"
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
        file_path = f"{cfg.log.mem_path}/replay_indices_{cfg.linear.target_task}.npy"
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


