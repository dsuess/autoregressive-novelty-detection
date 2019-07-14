import torch
from torch import nn
from torch.nn.functional import mse_loss


__all__ = ['AutoregressiveLoss', 'autoregressive_loss']



def autoregressive_loss(y, x, re_weight=0.5, reduction='mean', retlosses=False):
    autoreg_score, reconstruction = y
    autoreg_score = autoreg_score.sum()
    reconstruction_loss = mse_loss(x, reconstruction, reduction='sum')

    if reduction == 'mean':
        autoreg_score /= x.size(0)
        reconstruction_loss /= x.size(0)

    loss = (1 - re_weight) * autoreg_score + re_weight * reconstruction_loss

    if retlosses:
        return loss, {'reconstruction': reconstruction_loss,
                        'autoregressive': autoreg_score}
    else:
        return loss


class AutoregressiveLoss(nn.Module):

    def __init__(self, regressor, re_weight=.5, reduction='mean'):
        super().__init__()
        assert 0 <= re_weight <= 1
        assert reduction in {'mean', 'sum'}
        self.regressor = regressor
        self.register_scalar('re_weight', re_weight, dtype=torch.float32)
        self.reduction = reduction

    def forward(self, x, retlosses=False):
        y = self.regressor(x, return_reconstruction=True)
        return autoregressive_loss(
            y, x, retlosses=retlosses, re_weight=self.re_weight,
            reduction=self.reduction)
