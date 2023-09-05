import torch.nn.functional as F


def calc_loss(logits, targets, reduction='mean', class_weights=None, mode='ce'):
    logits = logits.view(1, -1)
    assert len(logits) == len(targets)
    # if mode == 'bce':
    #     return F.binary_cross_entropy_with_logits(logits, targets.float(), weight=class_weights, reduction=reduction)
    # elif mode == 'ce':

    return F.cross_entropy(logits, targets, weight=class_weights, reduction=reduction)
    # else:
    #     raise NotImplementedError

