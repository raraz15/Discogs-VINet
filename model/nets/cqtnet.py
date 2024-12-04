import torch
import torch.nn as nn

from utilities.tensor_op import l2_normalize

from .aux import IBN, GeM, Linear, SoftPool


class CQTNet(nn.Module):
    def __init__(
        self,
        ch_in: int,
        ch_out: int,
        norm: str = "bn",
        pool: str = "adaptive_max",
        l2_normalize: bool = True,
        projection: str = "linear",
    ):
        super().__init__()

        assert ch_in > 0
        assert ch_out > 0

        self.l2_normalize = l2_normalize

        if norm.lower() == "bn":
            norm = nn.BatchNorm2d
        elif norm.lower() == "ibn":
            norm = IBN
        elif norm.lower() == "in":
            norm = nn.InstanceNorm2d
        else:
            raise NotImplementedError

        self.front_end = nn.Sequential(
            nn.Conv2d(
                1,
                ch_in,
                kernel_size=(12, 3),
                dilation=(1, 1),
                padding=(6, 0),
                bias=False,
            ),
            norm(ch_in),
            nn.ReLU(inplace=True),
            #
            nn.Conv2d(
                ch_in, 2 * ch_in, kernel_size=(13, 3), dilation=(1, 2), bias=False
            ),
            norm(2 * ch_in),
            nn.ReLU(inplace=True),
            #
            nn.MaxPool2d((1, 2), stride=(1, 2), padding=(0, 1)),
            #
            nn.Conv2d(
                2 * ch_in, 2 * ch_in, kernel_size=(13, 3), dilation=(1, 1), bias=False
            ),
            norm(2 * ch_in),
            nn.ReLU(inplace=True),
            #
            nn.Conv2d(
                2 * ch_in, 2 * ch_in, kernel_size=(3, 3), dilation=(1, 2), bias=False
            ),
            norm(2 * ch_in),
            nn.ReLU(inplace=True),
            #
            nn.MaxPool2d((1, 2), stride=(1, 2), padding=(0, 1)),
            #
            nn.Conv2d(
                2 * ch_in, 4 * ch_in, kernel_size=(3, 3), dilation=(1, 1), bias=False
            ),
            norm(4 * ch_in),
            nn.ReLU(inplace=True),
            #
            nn.Conv2d(
                4 * ch_in, 4 * ch_in, kernel_size=(3, 3), dilation=(1, 2), bias=False
            ),
            norm(4 * ch_in),
            nn.ReLU(inplace=True),
            #
            nn.MaxPool2d((1, 2), stride=(1, 2), padding=(0, 1)),
            #
            nn.Conv2d(
                4 * ch_in, 8 * ch_in, kernel_size=(3, 3), dilation=(1, 1), bias=False
            ),
            norm(8 * ch_in),
            nn.ReLU(inplace=True),
            #
            nn.Conv2d(
                8 * ch_in, 8 * ch_in, kernel_size=(3, 3), dilation=(1, 2), bias=False
            ),
            norm(8 * ch_in),
            nn.ReLU(inplace=True),
            #
            nn.MaxPool2d((1, 2), stride=(1, 2), padding=(0, 1)),
            #
            nn.Conv2d(
                8 * ch_in, 16 * ch_in, kernel_size=(3, 3), dilation=(1, 1), bias=False
            ),
            nn.BatchNorm2d(16 * ch_in),
            nn.ReLU(inplace=True),
            #
            nn.Conv2d(
                16 * ch_in, 16 * ch_in, kernel_size=(3, 3), dilation=(1, 2), bias=False
            ),
            nn.BatchNorm2d(16 * ch_in),
            nn.ReLU(inplace=True),
        )

        if pool.lower() == "gem":
            self.pool = GeM()
        elif pool.lower() == "adaptive_max":
            self.pool = nn.AdaptiveMaxPool2d((1, 1))
        elif pool.lower() == "softpool":
            self.pool = SoftPool(16 * ch_in)
        else:
            raise NotImplementedError

        if projection.lower() == "linear":
            self.proj = Linear(16 * ch_in, ch_out, bias=False)
        elif projection.lower() == "affine":
            self.proj = Linear(16 * ch_in, ch_out)
        elif projection.lower() == "mlp":
            self.proj = nn.Sequential(
                Linear(16 * ch_in, 32 * ch_in),
                nn.ReLU(inplace=True),
                Linear(32 * ch_in, ch_out, bias=False),  # TODO bias=True?
            )
        elif projection.lower() == "none":
            self.proj = nn.Identity()
        else:
            raise NotImplementedError

        # TODO add batch norm here (If BNNeck)

    def forward(self, x):
        x = self.front_end(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.proj(x)

        # L2 normalization with 0 norm handling
        if self.l2_normalize:
            x = l2_normalize(x, precision="high")

        return x
