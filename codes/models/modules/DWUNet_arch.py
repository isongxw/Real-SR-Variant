import torch
import torch.nn as nn
import numpy as np
# import common

from pytorch_wavelets import DWTForward, DWTInverse


class _DCR_block(nn.Module):
    def __init__(self, channel_in):
        super(_DCR_block, self).__init__()

        self.conv_1 = nn.Conv2d(in_channels=channel_in, out_channels=int(channel_in/1.), kernel_size=3, stride=1, padding=1)

        self.conv_2 = nn.Conv2d(in_channels=int(channel_in*2.), out_channels=int(channel_in/1.), kernel_size=3, stride=1, padding=1)
       
        self.conv_3 = nn.Conv2d(in_channels=channel_in*3, out_channels=channel_in, kernel_size=3, stride=1, padding=1)
     

       # self.conv_1 = common.SEACB(in_channels=channel_in, out_channels=int(channel_in/2.))
        self.relu1 = nn.PReLU()
      #  self.conv_2 = common.SEACB(in_channels=int(channel_in*3/2.), out_channels=int(channel_in/2.))
        self.relu2 = nn.PReLU()
      #  self.conv_3 = common.SEACB(in_channels=channel_in*2, out_channels=channel_in)
        self.relu3 = nn.PReLU()

    def forward(self, x):
        residual = x

        out = self.relu1(self.conv_1(x))      # 1

        conc = torch.cat([x, out], 1)             # 2

        out = self.relu2(self.conv_2(conc))         #1

        conc = torch.cat([conc, out], 1)        #3

        out = self.relu3(self.conv_3(conc))  

        out = torch.add(out, residual)

        return out


class DWUNet(nn.Module):
    def __init__(self):
        super(DWUNet, self).__init__()
        self.upsample = nn.Upsample(scale_factor=4, mode='bicubic')

        self.DWT = DWTForward(J=1, wave='haar').cuda()     # 小波分解    
        self.IDWT = DWTInverse(wave='haar').cuda()       # 合成


        self.DCR_block21 = self.make_layer(_DCR_block, 3)
        self.DCR_block22 = self.make_layer(_DCR_block, 3)
        self.conv_i1 = nn.Conv2d(in_channels=12, out_channels=64, kernel_size=3, stride=1, padding=1)

        self.DCR_block31 = self.make_layer(_DCR_block, 64)
        self.DCR_block32 = self.make_layer(_DCR_block, 64)

        self.conv_i2 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1)

        self.DCR_block41 = self.make_layer(_DCR_block, 512)
        self.DCR_block42 = self.make_layer(_DCR_block, 512)

        self.conv_i3 = nn.Conv2d(in_channels=2048, out_channels=1024, kernel_size=3, stride=1, padding=1)


        self.DCR_block33 = self.make_layer(_DCR_block, 1280)
        self.DCR_block34 = self.make_layer(_DCR_block, 1280)
        self.conv_i4 = nn.Conv2d(in_channels=1280, out_channels=640, kernel_size=3, stride=1, padding=1)

        self.DCR_block23 = self.make_layer(_DCR_block, 224)
        self.DCR_block24 = self.make_layer(_DCR_block, 224)
        self.conv_i5 = nn.Conv2d(in_channels=224, out_channels=64, kernel_size=3, stride=1, padding=1)

        self.DCR_block13 = self.make_layer(_DCR_block, 16)
        self.DCR_block14 = self.make_layer(_DCR_block, 16)
        self.conv_f = nn.Conv2d(in_channels=16, out_channels=3, kernel_size=1, stride=1, padding=0)
        self.relu2 = nn.PReLU()

    def make_layer(self, block, channel_in):
        layers = []
        layers.append(block(channel_in))
        return nn.Sequential(*layers)

    def _transformer(self, DMT1_yl, DMT1_yh):
        list_tensor = []
        for i in range(3):
            list_tensor.append(DMT1_yh[0][:,:,i,:,:])
        list_tensor.append(DMT1_yl)
        return torch.cat(list_tensor, 1)

    def _Itransformer(self,out):
        yh = []
        C=int(out.shape[1]/4)
        # print(out.shape[0])
        y = out.reshape((out.shape[0], C, 4, out.shape[-2], out.shape[-1]))
        yl = y[:,:,0].contiguous()
        yh.append(y[:,:,1:].contiguous())
        return yl, yh


    def forward(self, x):
        x = self.upsample(x)
        # print(x.shape)
        residual = x               #   batchsize x 3x 64 x 64

#############   第一次分解         
        DMT1_yl,DMT1_yh = self.DWT(x)         #  batchsize x 3 x 32 x 32
        DMT1_yl = self.DCR_block21(DMT1_yl)
        DMT1_yl = self.DCR_block22(DMT1_yl)
        out = self._transformer(DMT1_yl, DMT1_yh)   # batchsize x 12 x 32 x 32
        conc2 =  self.conv_i1(out)                    #尺度对齐

#############   第二次分解
        DMT2_yl,DMT2_yh = self.DWT(conc2)           # 16 x 16
        DMT2_yl = self.DCR_block31(DMT2_yl)
        DMT2_yl = self.DCR_block32(DMT2_yl)
        out = self._transformer(DMT2_yl, DMT2_yh)    # 16 x 16
        conc3 =  self.conv_i2(out)


#############   第三次分解
        DMT3_yl,DMT3_yh = self.DWT(conc3)        # 8 x 8
        # print("DMT3_yl.shape, DMT3_yh.shape", len(DMT2_yl[0][0]), len(DMT2_yh[0][0]))
        DMT3_yl = self.DCR_block41(DMT3_yl)
        DMT3_yl = self.DCR_block42(DMT3_yl)
        # print("DMT3_yl.shape, DMT3_yh.shape", len(DMT2_yl[0][0]), len(DMT2_yh[0][0]))
        out = self._transformer(DMT3_yl, DMT3_yh)   # 8 x 8
        # print("conc4.shape, out.shape", out.shape)
        conc4 = self.conv_i3(out)
        # print("conc4.shape, out.shape", conc4.shape, out.shape)
        out = torch.cat([conc4, out], 1)
        # print("conc4.shape, out.shape", conc4.shape, out.shape)


#############   第一次合成
        out=self._Itransformer(out)
        out=self.IDWT(out)                    # 16 x 16
        # print("conc3.shape, out.shape", conc3.shape, out.shape)
        out = torch.cat([conc3, out], 1)
        out = self.DCR_block33(out)
        out = self.DCR_block34(out)
        out =  self.conv_i4(out)

#############   第二次合成
        out=self._Itransformer(out)
        out=self.IDWT(out)                    # 32 x 32
        out = torch.cat([conc2, out], 1)
        out = self.DCR_block23(out)
        out = self.DCR_block24(out)
        out =  self.conv_i5(out)

#############   第三次合成
        out=self._Itransformer(out)
        out=self.IDWT(out)            # 64 x 64
        out = self.DCR_block13(out)
        out = self.DCR_block14(out)
        out = self.relu2(self.conv_f(out))


        out = torch.add(residual, out)

        return out
