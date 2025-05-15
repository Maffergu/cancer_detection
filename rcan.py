import torch
import torch.nn as nn

# Bloque de atención de canal
class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.fc(y)
        return x * y

# Bloque residual + atención
class RCAB(nn.Module):
    def __init__(self, channel):
        super(RCAB, self).__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channel, channel, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, channel, 3, padding=1),
            CALayer(channel)
        )

    def forward(self, x):
        return self.body(x) + x

# RCAN básico
class RCAN(nn.Module):
    def __init__(self, num_blocks=5, channel=64, scale=2):
        super(RCAN, self).__init__()
        self.head = nn.Conv2d(1, channel, 3, padding=1)
        self.body = nn.Sequential(*[RCAB(channel) for _ in range(num_blocks)])
        self.upsample = nn.Sequential(
            nn.Conv2d(channel, channel * scale**2, 3, padding=1),
            nn.PixelShuffle(scale)
        )
        self.tail = nn.Conv2d(channel, 1, 3, padding=1)

    def forward(self, x):
        x = self.head(x)
        res = self.body(x)
        x = x + res
        x = self.upsample(x)
        x = self.tail(x)
        return x
