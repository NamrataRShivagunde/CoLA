import copy
import torch
from torch import nn
from typing import Iterable


class EmpiricalParamSmoothing(nn.Module):
    def __init__(
        self,
        base_model: nn.Module,
        n_smoothing: int = 1,
        gaussian_std: float = 0.1,
        smoothing_params: Iterable[str] = (),
    ):
        super(EmpiricalParamSmoothing, self).__init__()
        self.base_model = base_model
        self.n_smoothing = n_smoothing
        self.perturbed_model = copy.deepcopy(base_model)
        self.gaussian_std = gaussian_std
        self.smoothing_params = smoothing_params

    def perturb_model(self):
        for name, param in self.perturbed_model.named_parameters():
            if name in self.smoothing_params:
                param.add_(torch.randn_like(param.data, requires_grad=False) * self.gaussian_std)

    def forward(self, *args, **kwargs):
        self.perturb_model()
        output = self.base_model(*args, **kwargs)
        for _ in range(self.n_smoothing - 1):
            self.perturbed_model()
            output += self.base_model(*args, **kwargs)
        return output
