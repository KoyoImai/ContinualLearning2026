

import math
import logging
import numpy as np
import torch


from trainers.base import BaseLearner
from utils import AverageMeter



def adjust_learning_rate(cfg, optimizer, epoch):

    lr_enc = cfg.optimizer.learning_rate
    cosine = cfg.optimizer.scheduler.cosine
    lr_decay_rate = cfg.optimizer.scheduler.lr_decay_rate
    
    if cosine:
        eta_min_enc = lr_enc * (lr_decay_rate ** 3)
        lr_enc = eta_min_enc + (lr_enc - eta_min_enc) * (
                1 + math.cos(math.pi * epoch / cfg.train.epochs)) / 2
    else:
        steps = np.sum(epoch > np.asarray(cfg.optimizer.scheduler.lr_decay_epochs))
        if steps > 0:
            lr_enc = lr_enc * (lr_decay_rate ** steps)

    lr_list = [lr_enc]

    for idx, param_group in enumerate(optimizer.param_groups):
        # print('idx: ', idx)
        param_group['lr'] = lr_list[idx]


def warmup_learning_rate(cfg, epoch, batch_id, total_batches, optimizer):
    
    warm = cfg.optimizer.scheduler.warm
    warm_epochs = cfg.optimizer.scheduler.warm_epochs
    warmup_from_enc = cfg.optimizer.scheduler.warmup_from_enc
    warmup_to_enc = cfg.optimizer.scheduler.warmup_to_enc

    if warm and epoch <= warm_epochs:
        p = (batch_id + (epoch - 1) * total_batches) / \
            (warm_epochs * total_batches)
        lr_enc = warmup_from_enc + p * (warmup_to_enc - warmup_from_enc)
        lr_list = [lr_enc]

        for idx, param_group in enumerate(optimizer.param_groups):
            param_group['lr'] = lr_list[idx]




class CETrainer(BaseLearner):

    def __init__(self, cfg, model, model2, model_temp, criterion, optimizer, replay_indices, writer):
        super().__init__(cfg, model, model2, model_temp, criterion, optimizer, replay_indices, writer)

        # 蒸留タイプの決定
        self.distill_type = self.cfg.criterion.distill.type
    

    def train(self, train_loader, epoch):

        # model を train モードに変更
        self.model.train()
        self.model2.eval()

        # Averagemeter の初期化
        losses = AverageMeter()
        distill = AverageMeter()

        self.train_loader = train_loader

        # 学習率の調整
        adjust_learning_rate(self.cfg, self.optimizer, epoch)

        for idx, data in enumerate(self.train_loader):

            # 画像とラベルを取得
            images, labels = data

            # バッチサイズ
            bsz = labels.shape[0]

            # 画像とラベルを gpu に配置
            if torch.cuda.is_available():
                images = images[0].cuda(non_blocking=True)
                labels = labels.cuda(non_blocking=True)

            # warmup
            warmup_learning_rate(self.cfg, epoch, idx, len(train_loader), self.optimizer)

            # model にデータを入力
            output, encoded, feature = self.model(images)
            # print("output.shape: ", output.shape)

            # 損失計算
            loss = self.criterion(output, labels).mean()

            # 蒸留損失の計算
            loss_distill = self.distill(logits=output, encoded=encoded, feature=feature, images=images)
            distill.update(loss_distill.item(), bsz)

            loss += self.cfg.criterion.distill.power * loss_distill
            losses.update(loss.item(), bsz)

            # 現在の学習率
            current_lr = self.optimizer.param_groups[0]['lr']

            # 最適化ステップ
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # 学習記録の表示
            if (idx+1) % self.cfg.print_freq == 0 or idx+1 == len(self.train_loader):
                logging.info('Train: [{0}][{1}/{2}]\t'
                    'loss {loss.val:.3f} ({loss.avg:.3f})\t'
                    'distill {distill.val:.3f} ({distill.avg:.3f})\t'
                    'lr {lr:.5f}'.format(
                    epoch, idx + 1, len(self.train_loader), loss=losses, distill=distill, lr=current_lr)
                )
            # ===== TensorBoard logging =====
            if self.writer is not None:
                t = self.cfg.continual.target_task
                step = self.global_step
                self.writer.add_scalar(f"train/task{t}/loss", loss.item(), step)
                self.writer.add_scalar(f"train/task{t}/distill", loss_distill.item(), step)
                self.writer.add_scalar(f"train/task{t}/lr", current_lr, step)
                self.global_step += 1
                self.writer.flush()

    def distill(self, logits, encoded, feature, images):

        loss_distill = torch.tensor(0.)

        if self.distill_type == "l2_encoded":
            if self.cfg.continual.target_task > 0:

                # 過去モデルの出力を獲得
                with torch.no_grad():
                    _, pre_encoded, _ = self.model2(images)
                
                # 次元数
                D = pre_encoded.shape[1]

                # L2蒸留損失
                delta = encoded - pre_encoded
                loss_distill = (delta ** 2).sum(dim=1).mean()
        
        elif self.distill_type == "kd_logits":
            if self.cfg.continual.target_task > 0:

                # old classes の数
                old_classes = self.cfg.continual.target_task * self.cfg.continual.cls_per_task

                # 教師モデルの出力
                with torch.no_grad():
                    teacher_logits, _, _ = self.model2(images)

                # 生徒・教師の logits の形状を合わせる
                student_old = logits[:, :old_classes]
                teacher_old = teacher_logits[:, :old_classes]

        else:
            assert False
        
        return loss_distill

    def set_fc(self):

        target_task = self.cfg.continual.target_task
        cls_per_task = self.cfg.continual.cls_per_task
        
        # 現在タスクまでの合計クラス数
        nb_classes = (target_task + 1) * cls_per_task

        # self.model の fc層を拡張
        self.model.update_fc(nb_classes)
        self.model_temp.update_fc(nb_classes)

    def set_scheduler(self):

        if self.cfg.optimizer.scheduler.warm:

            cosine = self.cfg.optimizer.scheduler.cosine
            
            learning_rate = self.cfg.optimizer.learning_rate
            lr_decay_rate = self.cfg.optimizer.scheduler.lr_decay_rate

            epochs = self.cfg.train.epochs
            warm_epochs = self.cfg.optimizer.scheduler.warm_epochs

            if cosine:
                eta_min_encoder = learning_rate * (lr_decay_rate ** 3)
                self.cfg.optimizer.scheduler.warmup_to_enc = eta_min_encoder + (learning_rate - eta_min_encoder) * (
                        1 + math.cos(math.pi * warm_epochs / epochs)) / 2
            else:
                self.cfg.optimizer.scheduler.warmup_to_enc = learning_rate
                
