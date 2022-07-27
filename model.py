# coding: utf-8

import torch
from torch import nn
from torch.nn.parameter import Parameter


class G(nn.Module):
    def __init__(self):
        super(G, self).__init__()
        self.leaky_relu = nn.LeakyReLU(0.2)

        self.sn_conv1 = nn.utils.spectral_norm(nn.Conv2d(2, 256, kernel_size=3, stride=1, padding=1))
        self.bn1 = nn.BatchNorm2d(256, momentum=0.9, eps=1e-5)

        self.sn_conv2 = nn.utils.spectral_norm(nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1))
        self.bn2 = nn.BatchNorm2d(128, momentum=0.9, eps=1e-5)

        self.sn_conv3 = nn.utils.spectral_norm(nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1))
        self.bn3 = nn.BatchNorm2d(64, momentum=0.9, eps=1e-5)

        self.sn_conv4 = nn.utils.spectral_norm(nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1))
        self.bn4 = nn.BatchNorm2d(32, momentum=0.9, eps=1e-5)

        self.sn_conv5 = nn.utils.spectral_norm(nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0))

        for name, p in self.named_parameters():
            if 'bias' in name:
                nn.init.zeros_(p)
            elif 'bn' in name:
                nn.init.trunc_normal_(p, mean=1, std=1e-3)
            else:
                nn.init.trunc_normal_(p, std=1e-3)

    def forward(self, inf, vis):
        x = torch.cat([vis, inf], dim=1)
        x = self.leaky_relu(self.bn1(self.sn_conv1(x)))
        x = self.leaky_relu(self.bn2(self.sn_conv2(x)))
        x = self.leaky_relu(self.bn3(self.sn_conv3(x)))
        x = self.leaky_relu(self.bn4(self.sn_conv4(x)))
        return torch.tanh(self.sn_conv5(x))


class D(nn.Module):
    def __init__(self):
        super(D, self).__init__()
        self.leaky_relu = nn.LeakyReLU(0.2)

        self.sn_conv1 = nn.utils.spectral_norm(nn.Conv2d(1, 32, kernel_size=3, stride=2))

        self.sn_conv2 = nn.utils.spectral_norm(nn.Conv2d(32, 64, kernel_size=3, stride=2))
        self.bn1 = nn.BatchNorm2d(64, momentum=0.9, eps=1e-5)

        self.sn_conv3 = nn.utils.spectral_norm(nn.Conv2d(64, 128, kernel_size=3, stride=2))
        self.bn2 = nn.BatchNorm2d(128, momentum=0.9, eps=1e-5)

        self.sn_conv4 = nn.utils.spectral_norm(nn.Conv2d(128, 256, kernel_size=3, stride=2))
        self.bn3 = nn.BatchNorm2d(256, momentum=0.9, eps=1e-5)

        self.linear = nn.Linear(6 * 6 * 256, 1)

        for name, p in self.named_parameters():
            if 'bias' in name:
                nn.init.zeros_(p)
            elif 'bn' in name:
                nn.init.trunc_normal_(p, mean=1, std=1e-3)
            else:
                nn.init.trunc_normal_(p, std=1e-3)

    def forward(self, x):
        x = self.leaky_relu(self.sn_conv1(x))
        x = self.leaky_relu(self.bn1(self.sn_conv2(x)))
        x = self.leaky_relu(self.bn2(self.sn_conv3(x)))
        x = self.leaky_relu(self.bn3(self.sn_conv4(x)))
        x = x.flatten(start_dim=1)
        return self.linear(x)


class GAFNet(nn.Module):
    def __init__(self):
        super(GAFNet, self).__init__()
        self.leaky_relu = nn.LeakyReLU(0.2)

        self.sn_conv0 = nn.utils.spectral_norm(nn.Conv2d(2, 32, kernel_size=3, stride=1, padding=1))
        self.bn0 = nn.BatchNorm2d(32, momentum=0.9, eps=1e-5)

        self.sn_conv01 = nn.utils.spectral_norm(nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1))
        self.bn01 = nn.BatchNorm2d(64, momentum=0.9, eps=1e-5)

        self.sn_conv02 = nn.utils.spectral_norm(nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1))
        self.bn02 = nn.BatchNorm2d(128, momentum=0.9, eps=1e-5)

        # self.sn_conv1 = nn.utils.spectral_norm(nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1))
        # self.bn1 = nn.BatchNorm2d(256, momentum=0.9, eps=1e-5)

        self.attention_layer = Attention_fusion(128)

        # self.sn_conv2 = nn.utils.spectral_norm(nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1))
        # self.bn2 = nn.BatchNorm2d(128, momentum=0.9, eps=1e-5)

        self.sn_conv3 = nn.utils.spectral_norm(nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1))
        self.bn3 = nn.BatchNorm2d(64, momentum=0.9, eps=1e-5)

        self.sn_conv4 = nn.utils.spectral_norm(nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1))
        self.bn4 = nn.BatchNorm2d(32, momentum=0.9, eps=1e-5)

        self.sn_conv5 = nn.utils.spectral_norm(nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0))

        for name, p in self.named_parameters():
            if 'bias' in name:
                nn.init.zeros_(p)
            elif 'bn' in name:
                nn.init.trunc_normal_(p, mean=1, std=1e-3)
            else:
                nn.init.trunc_normal_(p, std=1e-3)

    def forward(self, inf, vis):
        x = torch.cat([vis, inf], dim=1)
        x = self.leaky_relu(self.bn0(self.sn_conv0(x)))
        x = self.leaky_relu(self.bn01(self.sn_conv01(x)))
        x = self.leaky_relu(self.bn02(self.sn_conv02(x)))
        # x = self.leaky_relu(self.bn1(self.sn_conv1(x)))

        x1 = self.attention_layer(x)

        # x1 = self.leaky_relu(self.bn2(self.sn_conv2(x1)))
        x1 = self.leaky_relu(self.bn3(self.sn_conv3(x1)))
        x1 = self.leaky_relu(self.bn4(self.sn_conv4(x1)))

        return torch.tanh(self.sn_conv5(x1))

class Attention_fusion(nn.Module):
    def __init__(self, channel, M = None):
        super(Attention_fusion, self).__init__()
        self.M = M
        self.channel = channel
        self.conv1 = nn.Sequential(
            nn.Conv2d(self.channel, self.channel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.channel),
            nn.LeakyReLU(0.2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(self.channel, self.channel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.channel),
            nn.LeakyReLU(0.2)
        )

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(channel, channel)
        self.sigmoid = nn.Sigmoid()

        self.conv3 = nn.Sequential(
            nn.Conv2d(self.channel, self.channel//2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.channel//2),
            nn.LeakyReLU(0.2)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(self.channel//2, 1, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(1),
        )


    def forward(self, x):
        # channel attention
        xn_0 = self.conv1(x)
        #xn = self.avg_pool(xn_0)
        xn = xn_0.mean(-1).mean(-1)
        xn = self.fc(xn)
        xn = xn.unsqueeze_(dim=-1).unsqueeze(dim=-1)
        xn = xn_0 * self.sigmoid(xn)

        # spatial attention
        xs_0 = self.conv2(x)
        xs = self.conv3(xs_0)
        xs = self.conv4(xs)
        xs = xs_0 * self.sigmoid(xs)

        return xs + xn


class SAFusion_layer(nn.Module):
    """Constructs a Channel Spatial Group module.

    Args:
        k_size: Adaptive selection of kernel size
    """

    def __init__(self, channel, groups=None):
        super(SAFusion_layer, self).__init__()
        self.groups = groups
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.cweight = Parameter(torch.zeros(1, channel // groups, 1, 1))
        self.cbias = Parameter(torch.ones(1, channel // groups, 1, 1))

        self.sweight = Parameter(torch.zeros(1, channel // (2 * groups), 1, 1))
        self.sbias = Parameter(torch.ones(1, channel // (2 * groups), 1, 1))

        self.sigmoid = nn.Sigmoid()
        self.gn = nn.GroupNorm(channel // groups, channel // groups)

        self.gnum = channel // groups
        self.conv1 = nn.Sequential(
                nn.Conv2d(self.gnum, self.gnum, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(self.gnum),
                nn.ReLU(inplace=False)
            )

        self.conv2 = nn.Sequential(
            nn.Conv2d(self.gnum, self.gnum, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.gnum),
            nn.ReLU(inplace=False)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(self.gnum, self.gnum, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.gnum),
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(2*self.gnum, self.gnum, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(self.gnum),
            nn.ReLU(inplace=False)
        )

    @staticmethod
    def channel_shuffle(x, groups):
        b, c, h, w = x.shape

        x = x.reshape(b, groups, -1, h, w)
        x = x.permute(0, 2, 1, 3, 4)

        # flatten
        x = x.reshape(b, -1, h, w)

        return x

    def forward(self, x1):
        b, c, h, w = x1.shape

        x1 = x1.reshape(b * self.groups, -1, h, w)

        # channel attention
        xn_0 = self.conv1(x1)
        xn = self.avg_pool(xn_0)
        xn = self.cweight * xn + self.cbias
        xn = xn_0 * self.sigmoid(xn)

        # spatial attention
        xs_0 = self.conv2(x1)
        xs = self.gn(xs_0)
        #xs = self.sweight * xs + self.sbias
        xs = self.conv3(xs)
        xs = xs_0 * self.sigmoid(xs)

        # concatenate along channel axis
        out = torch.cat([xn, xs], dim=1)
        out = self.conv4(out)
        out = out.reshape(b, -1, h, w)

        out = self.channel_shuffle(out, self.groups)
        return out
