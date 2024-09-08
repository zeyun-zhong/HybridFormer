# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""
This implementation is based on
https://github.com/rwightman/pytorch-image-models/blob/master/timm/data/mixup.py,
published under an Apache License 2.0.

COMMENT FROM ORIGINAL:
Mixup and Cutmix
Papers:
mixup: Beyond Empirical Risk Minimization (https://arxiv.org/abs/1710.09412)
CutMix: Regularization Strategy to Train Strong Classifiers with Localizable Features (https://arxiv.org/abs/1905.04899) # NOQA
Code Reference:
CutMix: https://github.com/clovaai/CutMix-PyTorch
Hacked together by / Copyright 2020 Ross Wightman
"""

import numpy as np
import torch


def convert_to_one_hot(targets, num_classes, on_value=1.0, off_value=0.0):
    """
    This function converts target class indices to one-hot vectors, given the
    number of classes.
    Args:
        targets (loader): Class labels.
        num_classes (int): Total number of classes.
        on_value (float): Target Value for ground truth class.
        off_value (float): Target Value for other classes.This value is used for
            label smoothing.
    """

    targets = targets.long().view(-1, 1)
    return torch.full(
        (targets.size()[0], num_classes), off_value, device=targets.device
    ).scatter_(1, targets, on_value)


def mixup_target(target, num_classes, lam=1.0, smoothing=0.0):
    """
    This function converts target class indices to one-hot vectors, given the
    number of classes.
    Args:
        targets (loader): Class labels.
        num_classes (int): Total number of classes.
        lam (float): lamba value for mixup/cutmix.
        smoothing (float): Label smoothing value.
    """
    off_value = smoothing / num_classes
    on_value = 1.0 - smoothing + off_value
    target1 = convert_to_one_hot(
        target,
        num_classes,
        on_value=on_value,
        off_value=off_value,
    )
    target2 = convert_to_one_hot(
        target.flip(0),
        num_classes,
        on_value=on_value,
        off_value=off_value,
    )
    return target1 * lam + target2 * (1.0 - lam)


class MixUp:
    """
    Apply mixup and/or cutmix for videos at batch level.
    mixup: Beyond Empirical Risk Minimization (https://arxiv.org/abs/1710.09412)
    CutMix: Regularization Strategy to Train Strong Classifiers with Localizable
        Features (https://arxiv.org/abs/1905.04899)
    """

    def __init__(
        self,
        mixup_alpha=0.8,
        mix_prob=1.0,
        label_smoothing=0.1,
        num_classes=60,
    ):
        """
        Args:
            mixup_alpha (float): Mixup alpha value.
            cutmix_alpha (float): Cutmix alpha value.
            mix_prob (float): Probability of applying mixup or cutmix.
            label_smoothing (float): Apply label smoothing to the mixed target
                tensor. If label_smoothing is not used, set it to 0.
            num_classes (int): Number of classes for target.
        """
        self.mixup_alpha = mixup_alpha
        self.mix_prob = mix_prob
        self.label_smoothing = label_smoothing
        self.num_classes = num_classes

    def _get_mixup_params(self):
        lam = 1.0
        if np.random.rand() < self.mix_prob:
            if self.mixup_alpha > 0.0:
                lam_mix = np.random.beta(self.mixup_alpha, self.mixup_alpha)
                lam = float(lam_mix)
        return lam

    def _mix_batch(self, x):
        lam = self._get_mixup_params()
        if lam == 1.0:
            return 1.0

        x_flipped = x.flip(0).mul_(1.0 - lam)
        x.mul_(lam).add_(x_flipped)
        return lam

    def __call__(self, x, target):
        assert len(x) > 1, "Batch size should be greater than 1 for mixup."
        lam = self._mix_batch(x)
        target = mixup_target(
            target, self.num_classes, lam, self.label_smoothing
        )
        return x, target
