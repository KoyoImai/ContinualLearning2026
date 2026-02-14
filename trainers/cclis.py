


from utils import AverageMeter
from trainers.base import BaseLearner


class CCLISLearner(BaseLearner):

    def __init__(self, cfg, model, model2, model_temp, criterion, optimizer, writer):
        super().__init__(cfg, model, model2, model_temp, criterion, optimizer, writer)

        # 蒸留タイプ
        self.distill_type = self.cfg.criterion.distill.type

        # その他のパラメータを初期化
        self.importance_weight = None