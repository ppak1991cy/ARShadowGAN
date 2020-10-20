"""
    ARShadowGAN
    func  : 网络结构
    Author: Chen Yu
    Date  : 2020.10.20
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def conv_block(in_dim, out_dim):
    return nn.Sequential(nn.Conv2d(in_dim, in_dim, kernel_size=3, stride=1, padding=1),
                         nn.ELU(True),
                         nn.Conv2d(in_dim, in_dim, kernel_size=3, stride=1, padding=1),
                         nn.ELU(True),
                         nn.Conv2d(in_dim, out_dim, kernel_size=1, stride=1, padding=0),
                         nn.AvgPool2d(kernel_size=2, stride=2))


def deconv_block(in_dim, out_dim):
    return nn.Sequential(nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=1, padding=1),
                         nn.ELU(True),
                         nn.Conv2d(out_dim, out_dim, kernel_size=3, stride=1, padding=1),
                         nn.ELU(True),
                         nn.UpsamplingNearest2d(scale_factor=2))


def blockUNet(in_c, out_c, name, transposed=False, bn=True, relu=True, dropout=False):
    block = nn.Sequential()
    if relu:
        block.add_module('%s_relu' % name, nn.ReLU(inplace=True))
    else:
        block.add_module('%s_leakyrelu' % name, nn.LeakyReLU(0.2, inplace=True))
    if not transposed:
        block.add_module('%s_conv' % name, nn.Conv2d(in_c, out_c, 4, 2, 1, bias=False))
    else:
        block.add_module('%s_tconv' % name, nn.ConvTranspose2d(in_c, out_c, 4, 2, 1, bias=False))
    if bn:
        block.add_module('%s_bn' % name, nn.BatchNorm2d(out_c))
    if dropout:
        block.add_module('%s_dropout' % name, nn.Dropout2d(0.5, inplace=True))
    return block


class Discriminator(nn.Module):
    def __init__(self, nc=3, ndf=32, hidden_size=96):
        super(Discriminator, self).__init__()

        self.conv1 = nn.Sequential(nn.Conv2d(nc, ndf, kernel_size=3, stride=1, padding=1), nn.ELU(True))
        self.conv2 = conv_block(ndf, ndf)
        self.conv3 = conv_block(ndf, ndf * 2)
        self.conv4 = conv_block(ndf * 2, ndf * 3)
        self.encode = nn.Conv2d(ndf * 3, hidden_size, kernel_size=1, stride=1, padding=0)
        self.decode = nn.Conv2d(hidden_size, ndf, kernel_size=1, stride=1, padding=0)
        self.deconv4 = deconv_block(ndf, ndf)
        self.deconv3 = deconv_block(ndf, ndf)
        self.deconv2 = deconv_block(ndf, ndf)
        self.deconv1 = nn.Sequential(nn.Conv2d(ndf, ndf, kernel_size=3, stride=1, padding=1),
                                     nn.ELU(True),
                                     nn.Conv2d(ndf, ndf, kernel_size=3, stride=1, padding=1),
                                     nn.ELU(True),
                                     nn.Conv2d(ndf, nc, kernel_size=3, stride=1, padding=1),
                                     nn.Tanh())

    def forward(self, x):
        out1 = self.conv1(x)
        out2 = self.conv2(out1)
        out3 = self.conv3(out2)
        out4 = self.conv4(out3)
        out5 = self.encode(out4)
        dout5 = self.decode(out5)
        dout4 = self.deconv4(dout5)
        dout3 = self.deconv3(dout4)
        dout2 = self.deconv2(dout3)
        dout1 = self.deconv1(dout2)
        return dout1


class Generator(nn.Module):
    def __init__(self, in_channel=3, output_nc=3, nf=32):
        super(Generator, self).__init__()

        layer_idx = 1
        name = 'layer%d' % layer_idx
        layer1 = nn.Sequential()
        layer1.add_module(name, nn.Conv2d(in_channel, nf, 4, 2, 1, bias=False))
        layer_idx += 1
        name = 'layer%d' % layer_idx
        layer2 = blockUNet(nf, nf * 2, name, transposed=False, bn=True, relu=False, dropout=False)
        layer_idx += 1
        name = 'layer%d' % layer_idx
        layer3 = blockUNet(nf * 2, nf * 4, name, transposed=False, bn=True, relu=False, dropout=False)
        layer_idx += 1
        name = 'layer%d' % layer_idx
        layer4 = blockUNet(nf * 4, nf * 8, name, transposed=False, bn=True, relu=False, dropout=False)
        layer_idx += 1
        name = 'layer%d' % layer_idx
        layer5 = blockUNet(nf * 8, nf * 8, name, transposed=False, bn=True, relu=False, dropout=False)
        layer_idx += 1

        name = 'dlayer%d' % layer_idx
        d_inc = nf * 4 * 2
        dlayer5 = blockUNet(d_inc, nf * 4, name, transposed=True, bn=True, relu=True, dropout=False)
        layer_idx -= 1
        name = 'dlayer%d' % layer_idx
        d_inc = nf * 4 * 3
        dlayer4 = blockUNet(d_inc, nf * 4, name, transposed=True, bn=True, relu=True, dropout=False)
        layer_idx -= 1
        name = 'dlayer%d' % layer_idx
        d_inc = nf * 4 * 2
        dlayer3 = blockUNet(d_inc, nf * 2, name, transposed=True, bn=True, relu=True, dropout=False)
        layer_idx -= 1
        name = 'dlayer%d' % layer_idx
        d_inc = nf * 2 * 2
        dlayer2 = blockUNet(d_inc, nf, name, transposed=True, bn=True, relu=True, dropout=False)
        layer_idx -= 1
        name = 'dlayer%d' % layer_idx
        dlayer1 = nn.Sequential()
        d_inc = nf * 2
        dlayer1.add_module('%s_relu' % name, nn.ReLU(inplace=True))
        dlayer1.add_module('%s_tconv' % name, nn.ConvTranspose2d(d_inc, output_nc, 4, 2, 1, bias=False))
        # dlayer1.add_module('%s_tanh' % name, nn.Tanh())

        self.refinement = RefinementNet(output_nc)

        self.layer1 = layer1
        self.layer2 = layer2
        self.layer3 = layer3
        self.layer4 = layer4
        self.layer5 = layer5
        self.dlayer5 = dlayer5
        self.dlayer4 = dlayer4
        self.dlayer3 = dlayer3
        self.dlayer2 = dlayer2
        self.dlayer1 = dlayer1

    def forward(self, x):
        out1 = self.layer1(x)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        out4 = self.layer4(out3)
        out5 = self.layer5(out4)
        dout5 = self.dlayer5(out5)
        dout5_out4 = torch.cat([dout5, out4], 1)
        dout4 = self.dlayer4(dout5_out4)
        dout4_out3 = torch.cat([dout4, out3], 1)
        dout3 = self.dlayer3(dout4_out3)
        dout3_out2 = torch.cat([dout3, out2], 1)
        dout2 = self.dlayer2(dout3_out2)
        dout2_out1 = torch.cat([dout2, out1], 1)
        dout1 = self.dlayer1(dout2_out1)

        ref = self.refinement(dout1)

        return F.tanh(dout1), F.tanh(ref)


class DenseBlock(nn.Module):
    def __init__(self, d_in, d_out):
        super(DenseBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(d_in)
        self.conv1 = nn.Conv2d(d_in, d_out, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(d_out)
        self.conv2 = nn.Conv2d(d_out, d_out, kernel_size=1, stride=1, padding=0)
        self.conv = nn.Conv2d(d_in + d_out, d_out, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        out = self.conv1(F.leaky_relu(self.bn1(x)))
        out = self.conv2(F.leaky_relu(self.bn2(out)))
        out = torch.cat([x, out], dim=1)
        out = self.conv(out)
        return out


class RefinementNet(nn.Module):
    def __init__(self, in_channel):
        super(RefinementNet, self).__init__()
        self.dense1 = nn.Sequential(
            DenseBlock(in_channel, 64),
            nn.BatchNorm2d(64)
        )
        self.dense2 = nn.Sequential(
            nn.LeakyReLU(0.2, inplace=True),
            DenseBlock(64, 128),
            nn.BatchNorm2d(128)
        )
        self.dense3 = nn.Sequential(
            nn.LeakyReLU(0.2, inplace=True),
            DenseBlock(128, 64),
            nn.BatchNorm2d(64)
        )
        self.dense4 = nn.Sequential(
            nn.LeakyReLU(0.2, inplace=True),
            DenseBlock(64, in_channel),
        )

    def forward(self, x):
        y = self.dense1(x)
        y = self.dense2(y)
        y = self.dense3(y)
        y = self.dense4(y)
        return F.tanh(y)









