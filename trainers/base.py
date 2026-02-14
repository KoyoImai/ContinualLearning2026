


import math
import logging
import numpy as np
from collections import defaultdict

import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler

# from utils import AverageMeter, adjust_learning_rate, warmup_learning_rate
from utils import AverageMeter
from models.resnet_cifar_co2l import LinearClassifier


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



    # tensorboardで可視化するための特徴
    def embedding(self, train_loader, replay_loader):
        pass


    #=============================
    # 線形分類による評価
    #=============================
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
                    # print('Train: [{0}][{1}/{2}]\t'
                    #     'loss {loss.val:.3f} ({loss.avg:.3f})'.format(
                    #     epoch, idx + 1, len(train_loader), loss=losses))
                    logging.info('Train: [{0}][{1}/{2}]\t'
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
                        # print('Test: [{0}/{1}]\t'
                        #     'Acc@1 {top1:.3f} {task_il:.3f}\t'
                        #     'lr {lr:.5f}'.format(
                        #         idx, len(val_loader),top1=np.sum(corr)/np.sum(cnt)*100., task_il=correct_task/np.sum(cnt)*100., lr=current_lr
                        #     ))
                        logging.info('Test: [{0}/{1}]\t'
                                     'Acc@1 {top1:.3f} {task_il:.3f}\t'
                                     'lr {lr:.5f}'.format(
                                            idx, len(val_loader),top1=np.sum(corr)/np.sum(cnt)*100., task_il=correct_task/np.sum(cnt)*100., lr=current_lr
                        ))
            # print(' * Acc@1 {top1:.3f} {task_il:.3f}'.format(top1=np.sum(corr)/np.sum(cnt)*100., task_il=correct_task/np.sum(cnt)*100.))
            logging.info(' * Acc@1 {top1:.3f} {task_il:.3f}'.format(top1=np.sum(corr)/np.sum(cnt)*100., task_il=correct_task/np.sum(cnt)*100.))


            # 学習率の調整
            scheduler.step()


    #=============================
    # NCM分類による評価
    #=============================
    def ncm_classify(self, val_encoded, val_labels, class_mean_encoded, metric="euclidean"):

        # クラス平均をまとめる
        class_ids = sorted(class_mean_encoded.keys())
        class_means = torch.stack([class_mean_encoded[c] for c in class_ids], dim=0)  # [num_classes, feature_dim]

        # 距離計算
        if metric == "euclidean":
            # ユークリッド距離
            # [N, C, D] → norm(dim=2) → [N, C]
            dists = torch.norm(val_encoded.unsqueeze(1) - class_means.unsqueeze(0), dim=2)

        elif metric == "cosine":
            # コサイン類似度 → 1 - cosine_similarityを距離として扱う
            # 正規化
            val_norm = torch.nn.functional.normalize(val_encoded, dim=1)      # [N, D]
            mean_norm = torch.nn.functional.normalize(class_means, dim=1)     # [C, D]
            # cos_sim: [N, C]
            cos_sim = torch.matmul(val_norm, mean_norm.T)
            dists = 1.0 - cos_sim  # 類似度が高いほど距離が小さくなる
        
        else:
            assert False
        
        # =========================================
        # 予測クラスを決定
        # =========================================
        pred_indices = torch.argmin(dists, dim=1)  # [N]
        pred_labels = torch.tensor([class_ids[i] for i in pred_indices], dtype=torch.long)

        # 精度計算
        correct = (pred_labels == val_labels).sum().item()
        accuracy = correct / len(val_labels)

        print(f"[{metric}] Nearest Class Mean Classification Accuracy: {accuracy:.4f} ({correct}/{len(val_labels)})")

        return pred_labels, accuracy
            


    def ncm_eval(self, train_loader, val_loader):

        target_task = self.cfg.ncm.target_task
        cls_per_task = self.cfg.continual.cls_per_task

        # model を eval モードに変更
        self.model.eval()

        # 現在データとリプレイバッファ内のデータから特徴を抽出
        train_encoded = defaultdict(list)

        with torch.no_grad():
            
            # 訓練用データの特徴量を取り出す
            for idx, (images, labels) in enumerate(train_loader):

                if torch.cuda.is_available():
                    images = images.cuda(non_blocking=True)
                    labels = labels.cuda(non_blocking=True)

                encoded = self.model.encoder(images)

                # features と encoded を辞書に格納する
                for enc, lbl in zip(encoded, labels):
                    train_encoded[int(lbl.item())].append(enc.detach().cpu())
        
            # 訓練用サンプルにおける各クラスの平均特徴を計算
            class_mean_encoded = {}

            for cls in train_encoded.keys():
                # torch.stack で [N, feature_dim] にまとめ、meanで平均を計算
                class_mean_encoded[cls] = torch.mean(torch.stack(train_encoded[cls]), dim=0)

            # 検証用データの特徴量を取り出す
            val_encoded = []
            val_labels = []

            for idx, (images, labels) in enumerate(val_loader):

                # gpu上に配置
                if torch.cuda.is_available():
                    images = images.cuda(non_blocking=True)
                    labels = labels.cuda(non_blocking=True)
                
                # 特徴量を取り出す
                encoded = self.model.encoder(images)

                # CPUに戻してリストに追加
                val_encoded.append(encoded.detach().cpu())
                val_labels.append(labels.detach().cpu())
            
            # 各バッチを結合して1つのテンソルにまとめる
            val_encoded = torch.cat(val_encoded, dim=0)     # shape: [num_val_samples, encoded_dim]
            val_labels = torch.cat(val_labels, dim=0)       # shape: [num_val_samples]

            print("=== 検証データ ===")
            print("val_encoded.shape:", val_encoded.shape)
            print("val_labels.shape:", val_labels.shape)

            # NCM分類を実行
            pred_labels_euclidean, acc_euclidean = self.ncm_classify(val_encoded, val_labels, class_mean_encoded, metric="euclidean")
            pred_labels_cosine, acc_cosine = self.ncm_classify(val_encoded, val_labels, class_mean_encoded, metric="cosine")


            # タスク増加での精度を計算
            task_acc_euclidean = []
            task_acc_cosine = []

            for taskid in range(target_task + 1):
                start_class = taskid * cls_per_task
                end_class   = (taskid + 1) * cls_per_task

                # 該当タスクの検証データを抽出
                mask = (val_labels >= start_class) & (val_labels < end_class)
                task_val_encoded = val_encoded[mask]
                task_val_labels  = val_labels[mask]

                if len(task_val_labels) == 0:
                    print(f"Task {taskid}: 検証データなし")
                    continue

                # 該当クラスの平均特徴を抽出
                task_class_mean_encoded = {cls: class_mean_encoded[cls] 
                                        for cls in range(start_class, end_class) 
                                        if cls in class_mean_encoded}

                # NCM分類を実行
                pred_euc, acc_euc = self.ncm_classify(task_val_encoded, task_val_labels, task_class_mean_encoded, metric="euclidean")
                pred_cos, acc_cos = self.ncm_classify(task_val_encoded, task_val_labels, task_class_mean_encoded, metric="cosine")

                task_acc_euclidean.append(acc_euc)
                task_acc_cosine.append(acc_cos)
                
                print(f"[Task {taskid}] Euclidean: {acc_euc:.4f}, Cosine: {acc_cos:.4f}")


        # 全体の平均精度
        mean_acc_euc = sum(task_acc_euclidean) / len(task_acc_euclidean)
        mean_acc_cos = sum(task_acc_cosine) / len(task_acc_cosine)

        
        print("=== Summary ===")
        print("Euclidean acc: ", acc_euclidean)
        print("Cosine acc: ", acc_cosine)
        print("Task-wise Euclidean acc:", task_acc_euclidean)
        print("Task-wise Cosine acc   :", task_acc_cosine)
        print("Mean Euclidean acc:", mean_acc_euc)
        print("Mean Cosine acc   :", mean_acc_cos)

        return acc_euclidean, acc_cosine