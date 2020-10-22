# coding=utf-8

import torch
from torch.nn.modules.loss import _Loss
import torch.nn.functional as F
from enum import IntEnum
import torch.nn as nn
from torch.autograd import Variable

class Criterion(_Loss):
    def __init__(self, alpha=1.0, name='criterion'):
        super().__init__()
        """Alpha is used to weight each loss term
        """
        self.alpha = alpha
        self.name = name

    def forward(self, input, target, weight=None, ignore_index=-1):
        """weight: sample weight
        """
        return

class FocalLoss(nn.Module):

    def __init__(self, gamma=2, weight=None, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.weight = weight.cuda()
        self.reduction = reduction

 

    def forward(self, output, target):
        # convert output to pseudo probability
        output = output.cuda()
        target = target.cuda()
        out_target = torch.stack([output[i, t] for i, t in enumerate(target)])
        probs = torch.sigmoid(out_target)
        focal_weight = torch.pow(1-probs, self.gamma)

        # add focal weight to cross entropy
        ce_loss = F.cross_entropy(output, target, weight=self.weight, reduction='none')
        focal_loss = focal_weight * ce_loss

        if self.reduction == 'mean':
            focal_loss = (focal_loss/focal_weight.sum()).sum()

        elif self.reduction == 'sum':
            focal_loss = focal_loss.sum()

        return focal_loss

class FocalCriterion(Criterion):
    def __init__(self, alpha=1.0, name='Focal Criterion'):
        super().__init__()
        self.alpha = alpha
        self.name = name
        self.weight = torch.tensor([0.7,0.3])
        self.focalloss = FocalLoss(2,self.weight)

    def forward(self, input, target, weight=None, ignore_index=-1):
        """weight: sample weight
        """
        loss = self.focalloss(input,target)
        return loss

class MultiFocalCriterion(Criterion):
    def __init__(self, alpha=1.0, name='Multi Focal Criterion'):
        super().__init__()
        self.alpha = alpha
        self.name = name
        self.weight = torch.tensor([6.3,16.1,13.8,14.5,10.4,6.1,20.2,13.5,39.2,15.2,14.5,51.7,13.2])
        self.focalloss = FocalLoss(2,self.weight)

    def forward(self, input, target, weight=None, ignore_index=-1):
        """weight: sample weight
        """
        loss = self.focalloss(input,target)
        return loss
    
class CeCriterion(Criterion):
    def __init__(self, alpha=1.0, name='Cross Entropy Criterion'):
        super().__init__()
        self.alpha = alpha
        self.name = name

    def forward(self, input, target, weight=None, ignore_index=-1):
        """weight: sample weight
        """
        if weight:
            loss = torch.mean(F.cross_entropy(input, target, reduce=False, ignore_index=ignore_index) * weight)
        else:
            loss = F.cross_entropy(input, target, ignore_index=ignore_index)
        loss = loss * self.alpha
        return loss

class SeqCeCriterion(CeCriterion):
    def __init__(self, alpha=1.0, name='Seq Cross Entropy Criterion'):
        super().__init__(alpha, name)

    def forward(self, input, target, weight=None, ignore_index=-1):
        if weight:
            loss = torch.mean(F.cross_entropy(input, target, reduce=False, ignore_index=ignore_index) * weight)
        else:
            tensor_input = torch.stack(input).transpose(0,1).transpose(1,2).cuda()
            target = target[:,1:]
            loss_cal = nn.NLLLoss(ignore_index=ignore_index)
            loss = loss_cal(tensor_input, target)
        loss = loss * self.alpha
        return loss

class MseCriterion(Criterion):
    def __init__(self, alpha=1.0, name='MSE Regression Criterion'):
        super().__init__()
        self.alpha = alpha
        self.name = name

    def forward(self, input, target, weight=None, ignore_index=-1):
        """weight: sample weight
        """
        if weight:
            loss = torch.mean(F.mse_loss(input.squeeze(), target, reduce=False) * 
                              weight.reshape((target.shape[0], 1)))
        else:
            loss = F.mse_loss(input.squeeze(), target)
        loss = loss * self.alpha
        return loss

class KlCriterion(Criterion):
    def __init__(self, alpha=1.0, name='KL Div Criterion'):
        super().__init__()
        self.alpha = alpha
        self.name = name

    def forward(self, input, target, weight=None, ignore_index=-1):
        """input/target: logits
        """
        input = input.float()
        target = target.float()
        loss = F.kl_div(F.log_softmax(input, dim=-1), F.softmax(target, dim=-1))
        loss = loss * self.alpha
        return loss

class SymKlCriterion(Criterion):
    def __init__(self, alpha=1.0, name='KL Div Criterion'):
        super().__init__()
        self.alpha = alpha
        self.name = name

    def forward(self, input, target, weight=None, ignore_index=-1):
        """input/target: logits
        """
        input = input.float()
        target = target.float()
        loss = F.kl_div(F.log_softmax(input, dim=-1), F.softmax(target.detach(), dim=-1)) + \
            F.kl_div(F.log_softmax(target, dim=-1), F.softmax(input.detach(), dim=-1))
        loss = loss * self.alpha
        return loss


class RankCeCriterion(Criterion):
    def __init__(self, alpha=1.0, name='Cross Entropy Criterion'):
        super().__init__()
        self.alpha = alpha
        self.name = name

    def forward(self, input, target, weight=None, ignore_index=-1, pairwise_size=1):
        input = input.view(-1, pairwise_size)
        target = target.contiguous().view(-1, pairwise_size)[:, 0]
        if weight:
            loss = torch.mean(F.cross_entropy(input, target, reduce=False, ignore_index=ignore_index) * weight)
        else:
            loss = F.cross_entropy(input, target, ignore_index=ignore_index)
        loss = loss * self.alpha
        return loss

class SpanCeCriterion(Criterion):
    def __init__(self, alpha=1.0, name='Span Cross Entropy Criterion'):
        super().__init__()
        """This is for extractive MRC, e.g., SQuAD, ReCoRD ... etc
        """
        self.alpha = alpha
        self.name = name

    def forward(self, input, target, weight=None, ignore_index=-1):
        """weight: sample weight
        """
        assert len(input) == 2
        start_input, end_input = input
        start_target, end_target = target
        if weight:
            b = torch.mean(F.cross_entropy(start_input, start_target, reduce=False, ignore_index=ignore_index) * weight)
            e = torch.mean(F.cross_entropy(end_input, end_target, reduce=False, ignore_index=ignore_index) * weight)
        else:
            b = F.cross_entropy(start_input, start_target, ignore_index=ignore_index)
            e = F.cross_entropy(end_input, end_target, ignore_index=ignore_index)
        loss = 0.5 * (b + e) * self.alpha
        return loss

class MlmCriterion(Criterion):
    def __init__(self, alpha=1.0, name='BERT pre-train Criterion'):
        super().__init__()
        self.alpha = alpha
        self.name = name

    def forward(self, input, target, weight=None, ignore_index=-1):
        """TODO: support sample weight, xiaodl
        """
        #import pdb; pdb.set_trace()
        mlm_y, y = target
        mlm_p, nsp_p = input
        mlm_p = mlm_p.view(-1, mlm_p.size(-1))
        mlm_y = mlm_y.view(-1)
        mlm_loss = F.cross_entropy(mlm_p, mlm_y, ignore_index=ignore_index)
        nsp_loss = F.cross_entropy(nsp_p, y)
        loss = mlm_loss + nsp_loss
        loss = loss * self.alpha
        return loss

class LossCriterion(IntEnum):
    CeCriterion = 0
    MseCriterion = 1
    RankCeCriterion = 2
    SpanCeCriterion = 3
    SeqCeCriterion = 4
    MlmCriterion = 5
    KlCriterion = 6
    SymKlCriterion = 7
    FocalCriterion = 8
    MultiFocalCriterion = 9

LOSS_REGISTRY = {
     LossCriterion.CeCriterion: CeCriterion,
     LossCriterion.MseCriterion: MseCriterion,
     LossCriterion.RankCeCriterion: RankCeCriterion,
     LossCriterion.SpanCeCriterion: SpanCeCriterion,
     LossCriterion.SeqCeCriterion: SeqCeCriterion,
     LossCriterion.MlmCriterion: MlmCriterion,
     LossCriterion.KlCriterion: KlCriterion,
     LossCriterion.SymKlCriterion: SymKlCriterion,
     LossCriterion.FocalCriterion: FocalCriterion,
     LossCriterion.MultiFocalCriterion: MultiFocalCriterion
}