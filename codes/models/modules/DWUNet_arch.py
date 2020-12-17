import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_wavelets import DWTForward, DWTInverse
from models.modules.PixelShuffleModel import PSUpsample


def unified_scale(x1, anchor):
    diffY = anchor.size()[2] - x1.size()[2]
    diffX = anchor.size()[3] - x1.size()[3]
    x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                    diffY // 2, diffY - diffY // 2])

    return x1


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class DCR_block(nn.Module):
    def __init__(self, channel_in):
        super(DCR_block, self).__init__()

        self.conv_1 = nn.Conv2d(
            in_channels=channel_in, out_channels=channel_in, kernel_size=3, stride=1, padding=1)
        self.conv_2 = nn.Conv2d(
            in_channels=channel_in * 2, out_channels=channel_in, kernel_size=3, stride=1, padding=1)
        self.conv_3 = nn.Conv2d(
            in_channels=channel_in * 3, out_channels=channel_in, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.PReLU()
        self.relu2 = nn.PReLU()
        self.relu3 = nn.PReLU()

    def forward(self, x):
        residual = x

        out = self.relu1(self.conv_1(x))
        conc = torch.cat([x, out], 1)

        out = self.relu2(self.conv_2(conc))
        conc = torch.cat([conc, out], 1)

        out = self.relu3(self.conv_3(conc))

        out = torch.add(out, residual)

        return out


class Down(nn.Module):
    def __init__(self, channel_in, channel_out):
        super(Down, self).__init__()

        self.wavelet = DWTForward(J=1, wave='haar')

        self.dcr = nn.Sequential(DCR_block(channel_in),
                                 DCR_block(channel_in))

        self.conv = nn.Conv2d(channel_in * 4, channel_out,
                              kernel_size=3, stride=1, padding=1)

    def transformer(self, DMT1_yl, DMT1_yh):
        list_tensor = []
        for i in range(3):
            list_tensor.append(DMT1_yh[0][:, :, i, :, :])
        list_tensor.append(DMT1_yl)
        return torch.cat(list_tensor, 1)

    def forward(self, x):
        yl, yh = self.wavelet(x)  # downsample channel_in
        yl = self.dcr(yl)  # channel_in
        out = self.transformer(yl, yh)  # channel_in * 4

        return self.conv(out)


class Up(nn.Module):
    def __init__(self, channel_in, channel_out):
        super(Up, self).__init__()

        self.wavelet_i = DWTInverse(wave='haar')
        self.dcr = nn.Sequential(DCR_block(channel_in),
                                 DCR_block(channel_in))
        self.conv1 = nn.Conv2d(channel_in // 4, channel_in //
                               2, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(channel_in, channel_out,
                               kernel_size=3, stride=1, padding=1)

    def transformer_i(self, out):
        yh = []
        C = int(out.shape[1] / 4)
        y = out.reshape((out.shape[0], C, 4, out.shape[-2], out.shape[-1]))
        yl = y[:, :, 0].contiguous()
        yh.append(y[:, :, 1:].contiguous())

        return yl, yh

    def forward(self, x1, x2):
        out = self.transformer_i(x2)  # channel_in // 4
        out = self.wavelet_i(out)
        out = self.conv1(out)
        out = torch.cat([unified_scale(out, x1), x1], 1)
        # out = torch.cat([x1, out], 1)
        out = self.dcr(out)

        return self.conv2(out)


class DWUNet(nn.Module):
    def __init__(self):
        super(DWUNet, self).__init__()

        # self.upsample1 = nn.Upsample(scale_factor=4, mode='bicubic')
        # self.convf = nn.Conv2d(in_channels=3, out_channels=3 * 16, kernel_size=3, stride=1, padding=1)
        # self.upsample = nn.PixelShuffle(upscale_factor=4)

        self.inc = DoubleConv(3, 64)
        self.down1 = Down(64, 128)  # 128*128
        self.down2 = Down(128, 256)  # 64*64
        self.down3 = Down(256, 512)  # 32*32

        self.up1 = Up(512, 256)  # 32*32
        self.up2 = Up(256, 128)  # 64*64
        self.up3 = Up(128, 64)  # 128*128

        self.outc = nn.Conv2d(in_channels=64, out_channels=3,
                              kernel_size=3, stride=1, padding=1)
        self.relu = nn.PReLU()

        self.upsample = PSUpsample()

    def forward(self, x):
        residual = x
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x = self.up1(x3, x4)
        x = self.up2(x2, x)
        x = self.up3(x1, x)
        out = self.outc(x)
        out = torch.add(self.relu(unified_scale(out, residual)), residual)
        out = self.upsample(out)
        # out = torch.add(self.relu(unified_scale(out, residual)), residual)
        # out = torch.sigmoid(self.upsample(self.convf(out)))

        return out


class DWUNet1(nn.Module):
    def __init__(self):
        super(DWUNet1, self).__init__()
        self.upsample = nn.Upsample(scale_factor=4, mode='bicubic')

        self.DWT = DWTForward(J=1, wave='haar').cuda()  # 小波分解
        self.IDWT = DWTInverse(wave='haar').cuda()  # 合成

        self.DCR_block21 = self.make_layer(DCR_block, 3)
        self.DCR_block22 = self.make_layer(DCR_block, 3)
        self.conv_i1 = nn.Conv2d(
            in_channels=12, out_channels=64, kernel_size=3, stride=1, padding=1)

        self.DCR_block31 = self.make_layer(DCR_block, 64)
        self.DCR_block32 = self.make_layer(DCR_block, 64)

        self.conv_i2 = nn.Conv2d(
            in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1)

        self.DCR_block41 = self.make_layer(DCR_block, 512)
        self.DCR_block42 = self.make_layer(DCR_block, 512)

        self.conv_i3 = nn.Conv2d(
            in_channels=2048, out_channels=1024, kernel_size=3, stride=1, padding=1)

        self.DCR_block33 = self.make_layer(DCR_block, 1280)
        self.DCR_block34 = self.make_layer(DCR_block, 1280)
        self.conv_i4 = nn.Conv2d(
            in_channels=1280, out_channels=640, kernel_size=3, stride=1, padding=1)

        self.DCR_block23 = self.make_layer(DCR_block, 224)
        self.DCR_block24 = self.make_layer(DCR_block, 224)
        self.conv_i5 = nn.Conv2d(
            in_channels=224, out_channels=64, kernel_size=3, stride=1, padding=1)

        self.DCR_block13 = self.make_layer(DCR_block, 16)
        self.DCR_block14 = self.make_layer(DCR_block, 16)
        self.conv_f = nn.Conv2d(
            in_channels=16, out_channels=3, kernel_size=1, stride=1, padding=0)
        self.relu2 = nn.PReLU()

    def make_layer(self, block, channel_in):
        layers = []
        layers.append(block(channel_in))
        return nn.Sequential(*layers)

    def _transformer(self, DMT1_yl, DMT1_yh):
        list_tensor = []
        for i in range(3):
            list_tensor.append(DMT1_yh[0][:, :, i, :, :])
        list_tensor.append(DMT1_yl)
        return torch.cat(list_tensor, 1)

    def _Itransformer(self, out):
        yh = []
        C = int(out.shape[1] / 4)
        # print(out.shape[0])
        y = out.reshape((out.shape[0], C, 4, out.shape[-2], out.shape[-1]))
        yl = y[:, :, 0].contiguous()
        yh.append(y[:, :, 1:].contiguous())
        return yl, yh

    def forward(self, x):
        x = self.upsample(x)
        # print(x.shape)
        residual = x  # batchsize x 3x 64 x 64

        # 第一次分解
        DMT1_yl, DMT1_yh = self.DWT(x)  # batchsize x 3 x 32 x 32
        DMT1_yl = self.DCR_block21(DMT1_yl)
        DMT1_yl = self.DCR_block22(DMT1_yl)
        out = self._transformer(DMT1_yl, DMT1_yh)  # batchsize x 12 x 32 x 32
        conc2 = self.conv_i1(out)  # 尺度对齐

        # 第二次分解
        DMT2_yl, DMT2_yh = self.DWT(conc2)  # 16 x 16
        DMT2_yl = self.DCR_block31(DMT2_yl)
        DMT2_yl = self.DCR_block32(DMT2_yl)
        out = self._transformer(DMT2_yl, DMT2_yh)  # 16 x 16
        conc3 = self.conv_i2(out)

        # 第三次分解
        DMT3_yl, DMT3_yh = self.DWT(conc3)  # 8 x 8
        # print("DMT3_yl.shape, DMT3_yh.shape", len(DMT2_yl[0][0]), len(DMT2_yh[0][0]))
        DMT3_yl = self.DCR_block41(DMT3_yl)
        DMT3_yl = self.DCR_block42(DMT3_yl)
        # print("DMT3_yl.shape, DMT3_yh.shape", len(DMT2_yl[0][0]), len(DMT2_yh[0][0]))
        out = self._transformer(DMT3_yl, DMT3_yh)  # 8 x 8
        # print("conc4.shape, out.shape", out.shape)
        conc4 = self.conv_i3(out)
        # print("conc4.shape, out.shape", conc4.shape, out.shape)
        out = torch.cat([conc4, out], 1)
        # print("conc4.shape, out.shape", conc4.shape, out.shape)

        # 第一次合成
        out = self._Itransformer(out)
        out = self.IDWT(out)  # 16 x 16
        # print("conc3.shape, out.shape", conc3.shape, out.shape)
        out = torch.cat([conc3, out], 1)
        out = self.DCR_block33(out)
        out = self.DCR_block34(out)
        out = self.conv_i4(out)

        # 第二次合成
        out = self._Itransformer(out)
        out = self.IDWT(out)  # 32 x 32
        out = torch.cat([conc2, out], 1)
        out = self.DCR_block23(out)
        out = self.DCR_block24(out)
        out = self.conv_i5(out)

        # 第三次合成
        out = self._Itransformer(out)
        out = self.IDWT(out)  # 64 x 64
        out = self.DCR_block13(out)
        out = self.DCR_block14(out)
        out = self.relu2(self.conv_f(out))

        out = torch.add(residual, out)

        return out
