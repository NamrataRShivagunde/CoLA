import torch
import torch.nn as nn
from transformers.activations import ACT2FN
import torch.nn.functional as F


import torch
import torch.nn as nn
from transformers.activations import ACT2FN


class ColaLayer(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        rank,
        bias=True,
        lr_act=True,
        lr_act_type="silu",
    ):
        super(ColaLayer, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank

        if lr_act:
            self.lr_act = ACT2FN[lr_act_type]

        target_sdv = (in_features + out_features) ** (-1 / 2)
        self.cola_a = nn.Parameter(
            torch.randn(in_features, rank) / rank ** (1 / 4) * target_sdv ** (1 / 2)
        )
        self.cola_b = nn.Parameter(
            torch.randn(rank, out_features) / rank ** (1 / 4) * target_sdv ** (1 / 2)
        )

        if bias == False:
            self.register_parameter("bias", None)
        else:
            stdv = 1.0 / out_features ** (1 / 2)
            self.bias = torch.nn.Parameter(torch.randn(out_features))
            self.bias.data.uniform_(-stdv, stdv)

    def extra_repr(self):
        return (
            f"cola_a: {self.cola_a.shape}, cola_b: {self.cola_b.shape}, "
            f"bias: {self.bias.shape if self.bias is not None else False}"
        )

    def forward(self, x):
        out = torch.matmul(x, self.cola_a)

        if hasattr(self, "lr_act"):
            out = self.lr_act(out)

        out = torch.matmul(out, self.cola_b)

        if self.bias is not None:
            out += self.bias

        return out


class ColaMDownProjLayer(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        rank,
        lr_act=True,
        lr_act_type="silu",
    ):
        super(ColaMDownProjLayer, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank

        if lr_act:
            self.lr_act = ACT2FN[lr_act_type]

        target_sdv = (in_features + out_features) ** (-1 / 2)
        self.cola_a = nn.Parameter(
            torch.randn(in_features, rank) / rank ** (1 / 4) * target_sdv ** (1 / 2)
        )

    def extra_repr(self):
        return f"cola_a: {self.cola_a.shape}"

    def forward(self, x):
        out = torch.matmul(x, self.cola_a)

        if hasattr(self, "lr_act"):
            out = self.lr_act(out)

        return out


class ColaMUpProjLayer(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        rank,
        bias=True,
    ):
        super(ColaMUpProjLayer, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank

        target_sdv = (in_features + out_features) ** (-1 / 2)
        self.cola_b = nn.Parameter(
            torch.randn(rank, out_features) / rank ** (1 / 4) * target_sdv ** (1 / 2)
        )
        if bias == False:
            self.register_parameter("bias", None)
        else:
            stdv = 1.0 / out_features ** (1 / 2)
            self.bias = torch.nn.Parameter(torch.randn(out_features))
            self.bias.data.uniform_(-stdv, stdv)

    def extra_repr(self):
        return f"cola_b: {self.cola_b.shape}, bias: {self.bias.shape if self.bias is not None else False}"

    def forward(self, x):
        out = torch.matmul(x, self.cola_b)

        if self.bias is not None:
            out += self.bias

        return out



# def orthonormalize(W):
#     # QR must run in float32 on GPU
#     W32 = W.float()
#     Q, _ = torch.linalg.qr(W32)  # orthonormal basis in FP32
#     return Q.to(W.dtype)  # cast back to bf16 if needed



# class ColaLayer(nn.Module):
#     """
#     W = (U Q) (P V)  = U (Q P) Vᵀ
#     where U and V are learned orthonormal subspaces (d x r),
#     and Z = Q P is a freely-learned r x r mixing matrix.
#     """
#     def __init__(self, in_features, out_features, rank, bias=True, lr_act=True, lr_act_type="silu"):
#         super().__init__()

#         self.in_features = in_features
#         self.out_features = out_features
#         self.rank = rank

#         # --- Learn subspace bases (will be re-orthonormalized in forward) ---
#         self.U = nn.Parameter(torch.randn(in_features, rank))
#         self.V = nn.Parameter(torch.randn(out_features, rank))

#         # --- Learn mixing matrix in latent r x r space ---
#         self.Z = nn.Parameter(torch.randn(rank, rank) * (1.0 / rank**0.5))

#         if lr_act:
#             self.lr_act = ACT2FN[lr_act_type]

#         if bias:
#             self.bias = nn.Parameter(torch.zeros(out_features))
#         else:
#             self.bias = None

#     def forward(self, x):
#         # Re-orthonormalize subspaces dynamically (key to rotating subspace)
#         U = orthonormalize(self.U)  # (d_in x r)
#         V = orthonormalize(self.V)  # (d_out x r)

#         # Effective low-rank weight
#         # W = U Z Vᵀ
#         weight = U @ self.Z @ V.t()  # (d_in x d_out)

#         out = x @ weight

#         if hasattr(self, "lr_act"):
#             out = self.lr_act(out)

#         if self.bias is not None:
#             out += self.bias

#         return out

#     def extra_repr(self):
#         return (
#             f"U: {self.U.shape}, V: {self.V.shape}, Z: {self.Z.shape}, "
#             f"bias: {self.bias.shape if self.bias is not None else False}"
#         )
