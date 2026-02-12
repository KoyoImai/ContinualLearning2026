

import math
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler


from utils import AverageMeter
from trainers.base import BaseLearner
from models.resnet_cifar_co2l import LinearClassifier


def adjust_learning_rate(cfg, optimizer, epoch):

    learning_rate = cfg.optimizer.learning_rate
    learning_rate_prototypes = cfg.optimizer.learning_rate_prototypes
    cosine = cfg.optimizer.scheduler.cosine
    lr_decay_rate = cfg.optimizer.scheduler.lr_decay_rate

    lr_enc = learning_rate
    lr_prot = learning_rate_prototypes
    
    if cosine:
        eta_min_enc = lr_enc * (lr_decay_rate ** 3)
        eta_min_prot = lr_prot * (lr_decay_rate ** 3)
        lr_enc = eta_min_enc + (lr_enc - eta_min_enc) * (
                1 + math.cos(math.pi * epoch / cfg.train.epochs)) / 2
        lr_prot = eta_min_prot + (lr_prot - eta_min_prot) * (
                1 + math.cos(math.pi * epoch / cfg.train.epochs)) / 2        
    else:
        steps = np.sum(epoch > np.asarray(cfg.optimizer.scheduler.lr_decay_epochs))
        if steps > 0:
            lr_enc = lr_enc * (lr_decay_rate ** steps)
            lr_prot = lr_prot * (lr_decay_rate ** steps)

    lr_list = [lr_enc, lr_enc, lr_prot]

    for idx, param_group in enumerate(optimizer.param_groups):
        # print('idx: ', idx)
        param_group['lr'] = lr_list[idx]


def warmup_learning_rate(cfg, epoch, batch_id, total_batches, optimizer):
    
    warm = cfg.optimizer.scheduler.warm
    warm_epochs = cfg.optimizer.scheduler.warm_epochs
    warmup_from_enc = cfg.optimizer.scheduler.warmup_from_enc
    warmup_from_prot = cfg.optimizer.scheduler.warmup_from_prot
    warmup_to_enc = cfg.optimizer.scheduler.warmup_to_enc
    warmup_to_prot = cfg.optimizer.scheduler.warmup_to_prot

    if warm and epoch <= warm_epochs:
        p = (batch_id + (epoch - 1) * total_batches) / \
            (warm_epochs * total_batches)
        lr_enc = warmup_from_enc + p * (warmup_to_enc - warmup_from_enc)
        lr_prot = warmup_from_prot + p * (warmup_to_prot - warmup_from_prot)
        lr_list = [lr_enc, lr_enc, lr_prot]

        for idx, param_group in enumerate(optimizer.param_groups):
            # print("lr_list[idx]: ", lr_list[idx])

            # print('idx: ', idx)
            param_group['lr'] = lr_list[idx]


class ProtoSupConTrainer(BaseLearner):

    def __init__(self, cfg, model, model2, criterion, optimizer):
        super().__init__(cfg, model, model2, criterion, optimizer)

        # 蒸留タイプの決定
        self.distill_type = self.cfg.criterion.distill.type


    def train(self, train_loader, epoch):

        # model を train モードに変更
        self.model.train()

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

            # プロトタイプの正規化
            with torch.no_grad():
                prev_task_mask = labels < self.cfg.continual.target_task * self.cfg.continual.cls_per_task

                w = self.model.prototypes.weight.data.clone()
                w = nn.functional.normalize(w, dim=1, p=2)
                self.model.prototypes.weight.copy_(w)
            
            # warm_up
            warmup_learning_rate(self.cfg, epoch, idx, len(train_loader), self.optimizer)

            encoded, features, output = self.model(images)
            # print("output.shape: ", output.shape)

            device = (torch.device('cuda')
                      if features.is_cuda
                      else torch.device('cpu'))
            self.device = device
            
            # 現在タスクのクラス
            target_labels = list(range(self.cfg.continual.target_task*self.cfg.continual.cls_per_task, (self.cfg.continual.target_task+1)*self.cfg.continual.cls_per_task))

            # =============================
            # 新しい知識獲得のための損失計算
            # =============================
            # プロトタイプベースの対照損失
            loss = self.criterion(output, features, labels, target_labels)

            # =============================
            # 蒸留損失の計算
            # =============================
            loss_distill = self.distill(features=features, images=images, labels=labels, output=output, target_labels=target_labels)
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
                print('Train: [{0}][{1}/{2}]\t'
                    'loss {loss.val:.3f} ({loss.avg:.3f})\t'
                    'distill {distill.val:.3f} ({distill.avg:.3f})\t'
                    'lr {lr:.5f}'.format(
                    epoch, idx + 1, len(self.train_loader), loss=losses, distill=distill, lr=current_lr))
        

    

    def distill(self, features, images, labels, output, target_labels):

        loss_distill = torch.tensor(0.)

        if self.distill_type == "prd":
            if self.cfg.continual.target_task > 0:

                # バッチに含まれるラベル一覧
                all_labels = torch.unique(labels).view(-1, 1)

                # 過去タスクのラベル一覧
                prev_all_labels = torch.arange(target_labels[0])

                # プロトタイプマスクを作成
                # （過去タスクのクラスに対応した出力のみを抽出可能）
                prototypes_mask = torch.scatter(
                    torch.zeros(len(prev_all_labels), self.cfg.continual.n_cls).float(),
                    1,
                    prev_all_labels.view(-1,1),
                    1
                ).to(self.device)

                # 過去タスクのサンプルだけを選別するマスク
                labels_mask = labels < min(target_labels)


                # ==================================
                # PRD (現在モデルの出力)
                # ==================================
                # 現在モデルで過去クラスに対応したプロトタイプの出力を計算
                sim_prev_task = torch.matmul(prototypes_mask, output)              # output から 過去クラスに対応した出力のみ取り出す
                features1_sim = torch.div(sim_prev_task, cfg.criterion.distill.current_temp)         # 温度パラメータで除算

                # 数値安定化
                logits_max1, _ = torch.max(features1_sim, dim=0, keepdim=True)
                features1_sim = features1_sim - logits_max1.detach()  # number stability

                row_size = features1_sim.size(0)
                # print("row_size: ", row_size)      # row_size:  2

                # logits を計算
                logits1 = torch.exp(features1_sim) / torch.exp(features1_sim).sum(dim=0, keepdim=True)


                # ==================================
                # PRD (過去モデルの出力)
                # ==================================              
                with torch.no_grad():
                    # 過去モデルで過去クラスに対応したプロトタイプの出力を計算
                    _, _, sim2_prev_task = self.model2(images)
                    sim2_prev_task = sim2_prev_task.T
                    sim2_prev_task = torch.matmul(prototypes_mask, sim2_prev_task)
                    features2_sim = torch.div(sim2_prev_task, self.cfg.criterion.distill.past_temp)

                    # 数値安定化
                    logits_max2, _ = torch.max(features2_sim, dim=0, keepdim=True)
                    features2_sim = features2_sim - logits_max2.detach()

                    # logits を計算
                    logits2 = torch.exp(features2_sim) / torch.exp(features2_sim).sum(dim=0, keepdim=True)
                

                # 蒸留損失を計算（KL-Divergence）
                loss_distill = (-logits2 * torch.log(logits1)).sum(0).mean()
                # print("loss_distill: ", loss_distill)

        elif self.distill_type is not None:
            loss_distill = torch.tensor(0.)
        else:
            assert False


        return loss_distill
    

    def set_scheduler(self):

        if self.cfg.optimizer.scheduler.warm:

            # warmup_from_enc = self.cfg.optimizer.scheduler.warmup_from_enc
            # warmup_from_prot = self.cfg.optimizer.scheduler.warmup_from_prot
            # warm_epochs = self.cfg.optimizer.scheduler.warm_epochs
            cosine = self.cfg.optimizer.scheduler.cosine
            
            learning_rate = self.cfg.optimizer.learning_rate
            lr_decay_rate = self.cfg.optimizer.scheduler.lr_decay_rate
            learning_rate_prototypes = self.cfg.optimizer.learning_rate_prototypes

            epochs = self.cfg.train.epochs
            warm_epochs = self.cfg.optimizer.scheduler.warm_epochs

            if cosine:
                eta_min_encoder = learning_rate * (lr_decay_rate ** 3)
                eta_min_prototypes = learning_rate_prototypes * (lr_decay_rate ** 3)
                self.cfg.optimizer.scheduler.warmup_to_enc = eta_min_encoder + (learning_rate - eta_min_encoder) * (
                        1 + math.cos(math.pi * warm_epochs / epochs)) / 2
                self.cfg.optimizer.scheduler.warmup_to_prot = eta_min_prototypes + (learning_rate_prototypes - eta_min_prototypes) * (
                        1 + math.cos(math.pi * warm_epochs / epochs)) / 2
            else:
                self.cfg.optimizer.scheduler.warmup_to_enc = learning_rate
                self.cfg.optimizer.scheduler.warmup_to_prot = learning_rate_prototypes



