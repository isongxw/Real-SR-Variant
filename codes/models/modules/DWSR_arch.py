import torch.nn as nn
from models.modules.wavelet import DWT_Haar, IWT_Haar


class Conv2dWithActivation(nn.Module):
  def __init__(self, in_channels, out_channels, kernel_size, padding=None, bias=True, activation=nn.ReLU(True)):
    super(Conv2dWithActivation, self).__init__()
    layers = []
    layers.append(nn.Conv2d(in_channels, out_channels, kernel_size,
                            padding=padding if padding is not None else kernel_size // 2, bias=bias))
    layers.append(activation)
    self.body = nn.Sequential(*layers)

  def forward(self, x):
    x = self.body(x)
    return x


class DWSR(nn.Module):
  def __init__(self, n_conv):
    super(DWSR, self).__init__()

    self.upsample = nn.Upsample(scale_factor=4, mode='bicubic')
    self.DWT = DWT_Haar()

    mid_layers = [Conv2dWithActivation(12, 64, 5)]
    for _ in range(n_conv):
      mid_layers.append(Conv2dWithActivation(64, 64, 3))
    mid_layers.append(Conv2dWithActivation(64, 12, 3))
    self.mid_layers = nn.Sequential(*mid_layers)

    self.IWT = IWT_Haar()

  def forward(self, x):
    x = self.upsample(x)

    lrsb = self.DWT(x)
    dsb = self.mid_layers(lrsb)
    srsb = 0.5 * lrsb + dsb
    y = self.IWT(srsb)
    return y.cuda()