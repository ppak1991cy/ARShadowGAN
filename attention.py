"""
    ARShadowGAN
    func  : Attention网络
    Author: Chen Yu
    Date  : 2020.10.20
"""
import torch.nn as nn
import torch.nn.init
import torch.nn.functional as F


class AttentionConv(nn.Module):
    r"""
        Self attention Layer.
        Source paper: https://arxiv.org/abs/1805.08318
    """

    def __init__(self, in_dim, activation=F.leaky_relu):
        super(AttentionConv, self).__init__()
        self.chanel_in = in_dim
        self.activation = activation

        self.f = nn.Conv2d(in_channels=in_dim, out_channels=1, kernel_size=1)
        self.g = nn.Conv2d(in_channels=in_dim, out_channels=1, kernel_size=1)
        self.h = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention feature maps

        """
        m_batchsize, C, width, height = x.size()

        f = self.f(x).view(m_batchsize, -1, width * height)  # B * (C//8) * (W * H)
        g = self.g(x).view(m_batchsize, -1, width * height)  # B * (C//8) * (W * H)
        h = self.h(x).view(m_batchsize, -1, width * height)  # B * C * (W * H)

        attention = torch.bmm(f.permute(0, 2, 1), g)  # B * (W * H) * (W * H)
        attention = self.softmax(attention)

        self_attetion = torch.bmm(h, attention)  # B * C * (W * H)
        self_attetion = self_attetion.view(m_batchsize, C, width, height)  # B * C * W * H

        out = self_attetion + x
        return out


class ResidualBlock(nn.Module):
    """ Residual block used to up sampling """

    def __init__(self, d_in, d_out):
        super(ResidualBlock, self).__init__()

        self.bn1 = nn.BatchNorm2d(d_in)
        self.conv1 = nn.Conv2d(d_in, d_out, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(d_out)
        self.conv2 = nn.Conv2d(d_out, d_out, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        out = self.conv1(F.leaky_relu(self.bn1(x)))
        out = self.conv2(F.leaky_relu(self.bn2(out)))
        out = out + x
        return out


class AttentionNet(nn.Module):

    def __init__(self, in_channel):
        super(AttentionNet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channel, 64, kernel_size=4, stride=2, padding=1),
            nn.AvgPool2d(2, 2),
            nn.BatchNorm2d(64),
            # nn.Dropout2d(0.5),
            AttentionConv(64),
            )
        self.conv2 = nn.Sequential(
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            # nn.Dropout2d(0.5),
            AttentionConv(128))
        self.conv3 = nn.Sequential(
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            # nn.Dropout2d(0.5)
        )

        self.res_block = nn.Sequential(
            ResidualBlock(128, 128),
            ResidualBlock(128, 128),
            ResidualBlock(128, 128),
        )

        self.shadow_dconv3 = nn.Sequential(
            nn.UpsamplingNearest2d(scale_factor=2),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            # nn.Dropout2d(0.5),
        )
        self.shadow_dconv2 = nn.Sequential(
            nn.UpsamplingNearest2d(scale_factor=2),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128 + 128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            # nn.Dropout2d(0.5),
        )
        self.shadow_dconv1 = nn.Sequential(
            nn.UpsamplingNearest2d(scale_factor=4),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128 + 64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            # nn.Dropout2d(0.5),
        )
        self.shadow_conv = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)

        self.occluder_dconv3 = nn.Sequential(
            nn.UpsamplingNearest2d(scale_factor=2),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            # nn.Dropout2d(0.5),
        )
        self.occluder_dconv2 = nn.Sequential(
            nn.UpsamplingNearest2d(scale_factor=2),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128 + 128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            # nn.Dropout2d(0.5),
        )
        self.occluder_dconv1 = nn.Sequential(
            nn.UpsamplingNearest2d(scale_factor=4),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128 + 64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            # nn.Dropout2d(0.5),
        )
        self.occluder_conv = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        feature_1 = self.conv1(x)  #128
        feature_2 = self.conv2(feature_1)  #64
        latent = self.conv3(feature_2)  #32
        latent = self.res_block(latent)

        shadow = self.shadow_dconv3(latent)
        shadow = self.shadow_dconv2(torch.cat([shadow, feature_2], dim=1))
        shadow = self.shadow_dconv1(torch.cat([shadow, feature_1], dim=1))
        shadow = self.shadow_conv(shadow)

        occluder = self.occluder_dconv3(latent)
        occluder = self.occluder_dconv2(torch.cat([occluder, feature_2], dim=1))
        occluder = self.occluder_dconv1(torch.cat([occluder, feature_1], dim=1))
        occluder = self.occluder_conv(occluder)

        return F.tanh(shadow), F.tanh(occluder)


if __name__ == '__main__':
    from dataset import ShadowARDataset

    dataset = ShadowARDataset()
    it = iter(dataset)
    sample = next(it)

    net = AttentionNet(1)
    res1, res2 = net(sample[3].unsqueeze(0))
    print(res1.shape)









