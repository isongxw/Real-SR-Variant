import torch.nn as nn


class Conv2dWithActivation(nn.Module):
    # 带激活函数的卷积层
    def __init__(self, in_channels, out_channels, kernel_size, padding=None, bias=True, activation=nn.ReLU()):
        super(Conv2dWithActivation, self).__init__()
        layers = []
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size,
                                padding=padding if padding is not None else kernel_size // 2, bias=bias))
        layers.append(activation)
        self.body = nn.Sequential(*layers)

    def forward(self, x):
        x = self.body(x)
        return x


class PSUpsample(nn.Module):
    def __init__(self):
        super(PSUpsample, self).__init__()
        self.conv1 = Conv2dWithActivation(3, 64, 3)
        self.conv2 = Conv2dWithActivation(64, 128, 3)
        self.conv3 = Conv2dWithActivation(128, 256, 3)
        self.conv4 = Conv2dWithActivation(256, 128, 3)
        self.conv5 = Conv2dWithActivation(128, 64, 3)
        self.conv6 = Conv2dWithActivation(64, 3*4**2, 3)
        self.act = nn.ReLU()
        self.ps = nn.PixelShuffle(4)

    def forward(self, x):
        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x))
        x = self.act(self.conv3(x))
        x = self.act(self.conv4(x))
        x = self.act(self.conv5(x))
        x = self.conv6(x)
        x = self.ps(x)
        return x
