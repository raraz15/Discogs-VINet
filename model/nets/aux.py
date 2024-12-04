import torch
import torch.nn as nn
import torch.nn.functional as F


class IBN(nn.Module):
    r"""Instance-Batch Normalization layer from
    `"Two at Once: Enhancing Learning and Generalization Capacities via IBN-Net"
    <https://arxiv.org/pdf/1807.09441.pdf>`
    Args:
        planes (int): Number of channels for the input tensor
    """

    def __init__(self, planes: int):
        super(IBN, self).__init__()

        self.ndim = planes // 2
        self.IN = nn.InstanceNorm2d(self.ndim, affine=True)
        self.BN = nn.BatchNorm2d(self.ndim)

    def forward(self, x):
        split = torch.split(x, self.ndim, 1)
        out1 = self.IN(split[0].contiguous())
        out2 = self.BN(split[1].contiguous())
        out = torch.cat((out1, out2), 1)
        return out


class GeM(nn.Module):
    def __init__(self, p=torch.log(torch.tensor(3)), eps=1e-6):
        super(GeM, self).__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        p = torch.exp(self.p)
        return F.avg_pool2d(x.clamp(min=self.eps).pow(p), (x.size(-2), x.size(-1))).pow(
            1.0 / p
        )

    def __repr__(self):
        return f"{self.__class__.__name__}(p={self.p.data.tolist()[0]:.4f}, eps={str(self.eps)})"


class Linear(torch.nn.Module):

    def __init__(self, nin, nout, dim=-1, bias=True):
        super().__init__()
        self.lin = torch.nn.Linear(nin, nout, bias=bias)
        self.dim = dim

    def forward(self, h):
        if self.dim != -1:
            h = h.transpose(self.dim, -1)
        h = self.lin(h)
        if self.dim != -1:
            h = h.transpose(self.dim, -1)
        return h


class SoftPool(torch.nn.Module):
    """Joan Serra"""

    def __init__(self, ncha):
        super().__init__()

        self.lin = Linear(ncha, 2 * ncha, dim=1, bias=False)
        self.norm = torch.nn.InstanceNorm1d(ncha, affine=True)
        self.flatten = torch.nn.Flatten(start_dim=2, end_dim=-1)

    def forward(self, h):
        h = self.flatten(h)
        h = self.lin(h)
        h, a = torch.chunk(h, 2, dim=1)
        a = torch.softmax(self.norm(a), dim=-1)
        return (h * a).sum(dim=-1)
