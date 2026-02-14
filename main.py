

import os
import copy
import hydra
import logging
import numpy as np


import torch
from torch.utils.tensorboard import SummaryWriter



from utils import seed_setup, save_model, make_dir
from models import make_model           # mode の作成
from criterions import make_criterion   # criterion の作成
from optimizers import make_optimizer   # optimizer の作成
from trainers import setup_trainer      # trainer の作成
from buffer import set_buffer           # buffer の作成
from dataloaders import set_loader      # dataloader の作成



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
    model2 = make_model(cfg=cfg)
    model_temp = make_model(cfg=cfg)

    #=======================
    # 損失関数の作成
    #=======================
    criterion = make_criterion(cfg)

    
    #=======================
    # Optimizer の作成
    #=======================
    optimizer = make_optimizer(cfg, model)


    #=======================
    # リプレイサンプルの初期化
    #=======================
    replay_indices = None

    #=======================
    # TensorBoard 
    #=======================
    cfg.log.tb_path = f"{cfg.log.base}/tb/"
    if not os.path.isdir(cfg.log.tb_path):
        os.makedirs(cfg.log.tb_path)
    writer = SummaryWriter(log_dir=f"{cfg.log.tb_path}")

    #=======================
    # trainerの作成
    #=======================
    trainer = setup_trainer(cfg, model, model2, model_temp, criterion, optimizer, writer)


    # タスク毎の学習エポック数
    original_epochs = cfg.train.epochs

    # 各タスクの学習
    for target_task in range(0, cfg.continual.n_task):

        # 現在タスクの更新
        cfg.continual.target_task = target_task
        # print('Start Training current task {}'.format(cfg.continual.target_task))
        logging.info('Start Training current task {}'.format(cfg.continual.target_task))

        # model2 のパラメータを model のパラメータでコピー
        trainer.model2 = copy.deepcopy(trainer.model)

        #=======================
        # リプレイサンプルの選択
        #=======================
        replay_indices = set_buffer(cfg, model, prev_indices=replay_indices)

        # バッファ内データのインデックスを保存（検証や分析時に読み込むため）
        np.save(
          os.path.join(cfg.log.mem_path, 'replay_indices_{target_task}.npy'.format(target_task=target_task)),
          np.array(replay_indices))

        #=======================
        # データローダの作成
        #=======================
        dataloader, vanila_loaders, subset_indices = set_loader(cfg, trainer, replay_indices)

        # subset_indices（このタスクで学習に使用する全てのデータのインデックス）を保存
        np.save(os.path.join(cfg.log.subset_path, 'subset_indices_{target_task}.npy'.format(target_task=target_task)), np.array(subset_indices))

        #=======================
        # 訓練開始前のその他の準備
        #=======================
        # 訓練前にエポック数を設定（初期エポックだけエポック数を変える場合に必要）
        if target_task == 0 and cfg.train.start_epochs is not None:
            cfg.train.epochs = cfg.train.start_epochs
        else:
            cfg.train.epochs = original_epochs

        # ランダム初期化のモデルを保存
        if target_task == 0:
            file_path = f"{cfg.log.model_path}/model_init.pth"
            save_model(model, optimizer, cfg,cfg.train.epochs, file_path)
        
        # scheduler 用の設定
        trainer.set_scheduler()

        
        #=======================
        # 訓練開始
        #=======================
        for epoch in range(1, cfg.train.epochs+1):

            # 学習を実行
            trainer.train(dataloader, epoch)

        # 保存（opt.model_path）
        file_path = f"{cfg.log.model_path}/model_{cfg.continual.target_task:02d}.pth"
        # save_model(model, method_tools["optimizer"], opt, opt.epochs, file_path)
        save_model(trainer.model, trainer.optimizer, cfg, cfg.train.epochs, file_path)


        #=======================
        # 分析用
        #=======================
        ckpt = torch.load(file_path, map_location='cpu')
        state_dict = ckpt['model']
        trainer.model_temp.load_state_dict(state_dict)

        trainer.embedding(vanila_loaders["train"], vanila_loaders["replay"])


if __name__ == "__main__":
    main()


