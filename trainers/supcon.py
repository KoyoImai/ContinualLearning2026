


import torch


from utils import AverageMeter, adjust_learning_rate, warmup_learning_rate
from trainers.base import BaseLearner


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
            # loss_distill = self.distill(features=features, images=images)

            # 特徴量を2viewに分割
            f1, f2 = torch.split(features, [bsz, bsz], dim=0)

            features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)

            # 損失計算
            loss = self.criterion(features, labels, target_labels=list(range(self.cfg.continual.target_task*self.cfg.continual.cls_per_task, (self.cfg.continual.target_task+1)*self.cfg.continual.cls_per_task)))
            print("loss: ", loss)

            # loss += self.cfg.criterion.distill.power * loss_distill
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
                    'lr {lr:.5f}'.format(
                    epoch, idx + 1, len(self.train_loader), loss=losses, lr=current_lr))
        
    
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