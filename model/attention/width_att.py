##############################################################################
# Created by: D. Alvarado-Carrillo
# Email: dora.alvarado@cimat.mx
#
# Note: This code was heavily inspired from https://github.com/junfu1115/DANet
##############################################################################
from __future__ import division
from torch.nn import Module, Conv2d, Parameter, Softmax
import torch
import torch.nn as nn

torch_ver = torch.__version__[:3]

__all__ = ['PAM', 'CAM']


class PAM(Module):
    """ Position attention module"""
    def __init__(self, in_dim, squeezing=8):
        super(PAM, self).__init__()
        self.chanel_in = in_dim

        self.query_conv = Conv2d(in_channels=in_dim, out_channels=in_dim // squeezing, kernel_size=1)
        self.key_conv = Conv2d(in_channels=in_dim, out_channels=in_dim // squeezing, kernel_size=1, dilation=2)
        self.value_conv = Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = Parameter(torch.zeros(1))
        self.softmax = Softmax(dim=-1)
    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value
                attention: B X (HxW) X (HxW)
        """
        m_batchsize, C, height, width = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width*height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width*height)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width*height)
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma*out
        return out


class CAM(Module):
    """ Channel attention module"""
    def __init__(self, in_dim):
        super(CAM, self).__init__()
        self.chanel_in = in_dim
        self.gamma = Parameter(torch.zeros(1))
        self.softmax = Softmax(dim=-1)

    def forward(self,x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value
                attention: B X C X C
        """
        m_batchsize, C, height, width = x.size()
        proj_query = x.view(m_batchsize, C, -1)
        n = proj_query.shape[-1]
        avg = torch.mean(proj_query, dim=2, keepdim=True).repeat([1,1,proj_query.shape[-1]])
        proj_query -=avg
        proj_key = proj_query.permute(0, 2, 1)
        energy = torch.bmm(1/n*proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = x.view(m_batchsize, C, -1)
        out = torch.bmm(attention, proj_value)
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma*out
        return out


class WAM(nn.Module):
    def __init__(self, in_channels, out_channels, squeezing_factor=4, squeezing_factor_pam=8, norm_layer=nn.BatchNorm2d):
        super(WAM, self).__init__()
        inter_channels = in_channels // squeezing_factor
        self.conv5a = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
                                    norm_layer(inter_channels),
                                    nn.ReLU())

        self.conv5c = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
                                    norm_layer(inter_channels),
                                    nn.ReLU())

        self.sa = PAM(inter_channels, squeezing=squeezing_factor_pam)
        self.sc = CAM(inter_channels)
        self.conv51 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
                                    norm_layer(inter_channels),
                                    nn.ReLU())
        self.conv52 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
                                    norm_layer(inter_channels),
                                    nn.ReLU())

        self.conv8 = nn.Sequential(nn.Dropout2d(0.1, False), nn.Conv2d(inter_channels, out_channels, 1))

    def forward(self, x):
        feat1 = self.conv5a(x)
        sa_feat = self.sa(feat1)
        sa_conv = self.conv51(sa_feat)

        feat2 = self.conv5c(x)
        sc_feat = self.sc(feat2)
        sc_conv = self.conv52(sc_feat)

        feat_sum = sa_conv + sc_conv

        sasc_output = self.conv8(feat_sum)

        return sasc_output

