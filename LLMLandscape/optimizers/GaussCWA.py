import torch
from torch import nn
import copy
from torch.optim import Optimizer, AdamW
from typing import Iterable

__all__ = ["GaussianAugmentedOptimizer"]

"""
使用这些优化器，一定一定注意一点：
训练前调用before_training()
训练后调用after_training()
"""


class GaussianAugmentedOptimizer(Optimizer):
    def __init__(
        self,
        base_optimizer: Optimizer,
        model: nn.Module,
        gaussian_std: float = 1e-3,
        excluded_param_names: Iterable[str] = (),
        backup_device=torch.device("cuda"),
    ):
        super(GaussianAugmentedOptimizer, self).__init__(base_optimizer.param_groups, base_optimizer.defaults)
        self.gaussian_std = gaussian_std
        self.real_params = dict()
        self.model = model
        self.excluded_param_names = excluded_param_names
        self.base_optimizer = base_optimizer
        self.backup_device = backup_device
        self.count = 0
        self.before_training()

    @torch.no_grad()
    def backup_real_params(self):
        self.real_params.clear()
        self.real_params = {
            k: copy.deepcopy(v.data.to(self.backup_device))
            for k, v in self.model.named_parameters()
            if k not in self.excluded_param_names
        }

    @torch.no_grad()
    def perturb_model(self):
        for name, param in self.model.named_parameters():
            if name not in self.excluded_param_names:
                param.add_(self.gaussian_std * torch.randn_like(param.data, requires_grad=False))

    @torch.no_grad()
    def recover_model(self):
        for name, param in self.model.named_parameters():
            if name not in self.excluded_param_names:
                param.mul_(0).add_(self.real_params[name].to(param.device))

    @torch.no_grad()
    def before_training(self):  # Call this function before training
        print("DEBUG: before_training was called")
        if self.count > 0:
            self.recover_model()
        self.backup_real_params()
        self.perturb_model()

    @torch.no_grad()
    def after_training(self):  # Call this function after training
        print("DEBUG: after_training was called")
        self.recover_model()

    @torch.no_grad()
    def step(self, *args, **kwargs):
        self.recover_model()
        self.base_optimizer.step(*args, **kwargs)
        self.backup_real_params()
        self.count += 1
        if self.count % 10 == 0:
            print("steps: ", self.count)
        self.perturb_model()
