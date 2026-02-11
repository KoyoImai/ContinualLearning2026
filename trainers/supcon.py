

import math
import numpy as np

import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler


# from utils import AverageMeter, adjust_learning_rate, warmup_learning_rate
from utils import AverageMeter
from trainers.base import BaseLearner
from models.resnet_cifar_co2l import LinearClassifier


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



class SupConTrainer(BaseLearner):

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

            # warm_up
            # if self.cfg.continual.target_task > 0:
            #     warmup_learning_rate(self.cfg, epoch, idx, len(train_loader), self.optimizer)
            warmup_learning_rate(self.cfg, epoch, idx, len(train_loader), self.optimizer)

            # ラベルあり2viewの画像を結合
            images = torch.cat([images[0], images[1]], dim=0)

            # gpu上に配置
            if torch.cuda.is_available():
                images = images.cuda(non_blocking=True)
            

            with torch.no_grad():
                prev_task_mask = labels < self.cfg.continual.target_task * self.cfg.continual.cls_per_task
                prev_task_mask = prev_task_mask.repeat(2) 
            

            # modelにデータを入力
            features, encoded = self.model(images, return_feat=True)

            # 蒸留損失の計算
            loss_distill = self.distill(features=features, images=images)
            distill.update(loss_distill.item(), bsz)

            # 特徴量を2viewに分割
            f1, f2 = torch.split(features, [bsz, bsz], dim=0)

            features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)

            # 損失計算
            # loss = self.criterion(features, labels, target_labels=list(range(self.cfg.continual.target_task*self.cfg.continual.cls_per_task, (self.cfg.continual.target_task+1)*self.cfg.continual.cls_per_task)))
            loss = self.criterion(features, labels)
            # print("loss: ", loss)

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
        
    
    def distill(self, features, images):

        loss_distill = torch.tensor(0.)
        # print("self.distill_type: ", self.distill_type)

        if self.distill_type == "ird":
            if self.cfg.continual.target_task > 0:
                features1_prev_task = features

                features1_sim = torch.div(torch.matmul(features1_prev_task, features1_prev_task.T), self.cfg.criterion.distill.current_temp)
                logits_mask = torch.scatter(
                    torch.ones_like(features1_sim),
                    1,
                    torch.arange(features1_sim.size(0)).view(-1, 1).cuda(non_blocking=True),
                    0
                )
                logits_max1, _ = torch.max(features1_sim * logits_mask, dim=1, keepdim=True)
                features1_sim = features1_sim - logits_max1.detach()
                row_size = features1_sim.size(0)
                logits1 = torch.exp(features1_sim[logits_mask.bool()].view(row_size, -1)) / torch.exp(features1_sim[logits_mask.bool()].view(row_size, -1)).sum(dim=1, keepdim=True)

                with torch.no_grad():
                    features2_prev_task = self.model2(images)

                    features2_sim = torch.div(torch.matmul(features2_prev_task, features2_prev_task.T), self.cfg.criterion.distill.past_temp)
                    logits_max2, _ = torch.max(features2_sim*logits_mask, dim=1, keepdim=True)
                    features2_sim = features2_sim - logits_max2.detach()
                    logits2 = torch.exp(features2_sim[logits_mask.bool()].view(row_size, -1)) /  torch.exp(features2_sim[logits_mask.bool()].view(row_size, -1)).sum(dim=1, keepdim=True)
                    # print('logits2.shape: ', logits2.shape)  # logits2.shape:  torch.Size([1024, 1023])

                loss_distill = (-logits2 * torch.log(logits1)).sum(1).mean()
        
        elif self.distill_type is not None:
            loss_distill = torch.tensor(0.)
        else:
            assert False


        return loss_distill


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
                

    def linear_eval(self, train_loader, val_loader):

        # classifierの準備
        classifier = LinearClassifier(name="resnet18", num_classes=self.cfg.continual.n_cls, seed=self.cfg.seed)
        if torch.cuda.is_available():
            classifier = classifier.cuda()
        
        # classifierのOptimizer
        optimizer = optim.SGD(classifier.parameters(),
                            lr=self.cfg.linear.train.learning_rate,
                            momentum=self.cfg.linear.train.momentum,
                            weight_decay=self.cfg.linear.train.weight_decay)

        # schedulerの設定
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[60, 75, 90], gamma=0.2)

        # 損失関数の作成
        criterion = torch.nn.CrossEntropyLoss()

        for epoch in range(1, self.cfg.linear.train.epochs):

            # modelをevalモード，classifierをtrainモードに変更
            self.model.eval()
            classifier.train()
            
            losses = AverageMeter()

            # 1エポック分の学習
            for idx, (images, labels) in enumerate(train_loader):

                images = images.cuda(non_blocking=True)
                labels = labels.cuda(non_blocking=True)
                bsz = labels.shape[0]

                # 特徴量獲得
                with torch.no_grad():
                    features = self.model.encoder(images)
                output = classifier(features.detach())
                loss = criterion(output, labels)

                # update metric
                losses.update(loss.item(), bsz)
                # cnt += bsz

                # 最適化ステップ
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # 現在の学習率
                current_lr = optimizer.param_groups[0]['lr']

                # 学習記録の表示
                if (idx+1) % self.cfg.print_freq == 0 or idx+1 == len(train_loader):
                    print('Train: [{0}][{1}/{2}]\t'
                        'loss {loss.val:.3f} ({loss.avg:.3f})'.format(
                        epoch, idx + 1, len(train_loader), loss=losses))
                

            # 検証（これまでの全てのタスクを使用）
            self.model.eval()
            classifier.eval()

            losses = AverageMeter()

            corr = [0.] * (self.cfg.linear.target_task + 1) * self.cfg.continual.cls_per_task
            cnt  = [0.] * (self.cfg.linear.target_task + 1) * self.cfg.continual.cls_per_task
            correct_task = 0.0

            with torch.no_grad():
                for idx, (images, labels) in enumerate(val_loader):
                    images = images.float().cuda()
                    labels = labels.cuda()
                    bsz = labels.shape[0]

                    # forward
                    output = classifier(self.model.encoder(images))
                    loss = criterion(output, labels)

                    # update metric
                    losses.update(loss.item(), bsz)

                    #
                    cls_list = np.unique(labels.cpu())
                    correct_all = (output.argmax(1) == labels)

                    for tc in cls_list:
                        mask = labels == tc
                        correct_task += (output[mask, (tc // self.cfg.continual.cls_per_task) * self.cfg.continual.cls_per_task : ((tc // self.cfg.continual.cls_per_task)+1) * self.cfg.continual.cls_per_task].argmax(1) == (tc % self.cfg.continual.cls_per_task)).float().sum()

                    for c in cls_list:
                        mask = labels == c
                        corr[c] += correct_all[mask].float().sum().item()
                        cnt[c] += mask.float().sum().item()
                    
                    if (idx+1) % self.cfg.print_freq == 0 or idx+1 == len(val_loader):
                        print('Test: [{0}/{1}]\t'
                            'Acc@1 {top1:.3f} {task_il:.3f}\t'
                            'lr {lr:.5f}'.format(
                                idx, len(val_loader),top1=np.sum(corr)/np.sum(cnt)*100., task_il=correct_task/np.sum(cnt)*100., lr=current_lr
                            ))
            print(' * Acc@1 {top1:.3f} {task_il:.3f}'.format(top1=np.sum(corr)/np.sum(cnt)*100., task_il=correct_task/np.sum(cnt)*100.))

            # 学習率の調整
            scheduler.step()




