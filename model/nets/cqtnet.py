import torch
import torch.nn as nn


class CQTNet(nn.Module):
    def __init__(self, l2_normalize=True):
        super().__init__()

        self.l2_normalize = l2_normalize

        self.features = nn.Sequential(
            nn.Conv2d(
                1, 32, kernel_size=(12, 3), dilation=(1, 1), padding=(6, 0), bias=False
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            #
            nn.Conv2d(32, 64, kernel_size=(13, 3), dilation=(1, 2), bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            #
            nn.MaxPool2d((1, 2), stride=(1, 2), padding=(0, 1)),
            #
            nn.Conv2d(64, 64, kernel_size=(13, 3), dilation=(1, 1), bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            #
            nn.Conv2d(64, 64, kernel_size=(3, 3), dilation=(1, 2), bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            #
            nn.MaxPool2d((1, 2), stride=(1, 2), padding=(0, 1)),
            #
            nn.Conv2d(64, 128, kernel_size=(3, 3), dilation=(1, 1), bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            #
            nn.Conv2d(128, 128, kernel_size=(3, 3), dilation=(1, 2), bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            #
            nn.MaxPool2d((1, 2), stride=(1, 2), padding=(0, 1)),
            #
            nn.Conv2d(128, 256, kernel_size=(3, 3), dilation=(1, 1), bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            #
            nn.Conv2d(256, 256, kernel_size=(3, 3), dilation=(1, 2), bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            #
            nn.MaxPool2d((1, 2), stride=(1, 2), padding=(0, 1)),
            #
            nn.Conv2d(256, 512, kernel_size=(3, 3), dilation=(1, 1), bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            #
            nn.Conv2d(512, 512, kernel_size=(3, 3), dilation=(1, 2), bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )

        self.pool = nn.AdaptiveMaxPool2d((1, 1))
        self.proj = nn.Sequential(
            nn.Linear(512, 512, bias=False),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.proj(x)

        if self.l2_normalize:
            # L2 normalization
            norms = torch.norm(x, p=2, dim=1, keepdim=True)
            # Create a mask for zero norm entries
            mask = (norms == 0).type_as(norms)
            # Add a small value to zero norm entries
            norms = norms + (mask * 1e-12)
            # Normalize the embeddings
            x = x / norms

        return x
