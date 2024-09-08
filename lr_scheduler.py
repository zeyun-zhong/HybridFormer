import torch
import math


class WarmUpCosineAnnealingLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, T_max, warmup_epochs, eta_min=0,
                 last_epoch=-1, verbose=False):
        self.T_max = T_max
        self.warmup_epochs = warmup_epochs
        self.eta_min = eta_min
        super(WarmUpCosineAnnealingLR, self).__init__(optimizer, last_epoch,
                                                      verbose)

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            # Linearly increase the learning rate
            lr_scale = self.last_epoch / self.warmup_epochs
            return [max(base_lr * lr_scale, self.eta_min) for base_lr in self.base_lrs]

        else:
            # Cosine annealing
            epoch_adj = self.last_epoch - self.warmup_epochs
            return [
                self.eta_min + (base_lr - self.eta_min) *
                (1 + math.cos(math.pi * epoch_adj / (
                            self.T_max - self.warmup_epochs))) / 2
                for base_lr in self.base_lrs
            ]