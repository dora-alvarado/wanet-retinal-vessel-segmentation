########################################################################################################################
# This is a pytorch implementation of the WA-Net architecture, from the preprint paper:
# [https://....]
########################################################################################################################
import torch
import torch.nn as nn
from scipy.ndimage.filters import gaussian_filter

from .wanet_parts import UNet
from .attention.selector import get_att_layer
from .gmf_filters import ApplyDGMF

class WANet(nn.Module):
    def __init__(self, n_channels, n_classes=2, n_dgmf=32, base_model=UNet, \
                 dgmf_kernel_size=(7, 7), max_sigma=3., min_sigma=0.10):
        super().__init__()
        self.n_classes = n_classes
        self.n_dgmf = n_dgmf
        self.base_model = base_model
        self.att_layer = get_att_layer("WAM")
        self.dropout2d = nn.Dropout2d(p=0.15)
        #
        self.dgmf_kernel_size = dgmf_kernel_size
        step = (max_sigma - min_sigma) / n_dgmf
        self.sigmas = nn.Parameter(torch.arange(min_sigma, max_sigma, step), requires_grad=False)
        self.alphas = nn.Parameter(torch.zeros(n_dgmf), requires_grad=True)
        self.dgmf_order = 2
        dx = torch.tensor(
            gaussian_filter((torch.rand(*self.dgmf_kernel_size) * 2 - 1), 4., mode="constant", cval=0))  # .to(x.device)
        dy = torch.tensor(gaussian_filter((torch.rand(*self.dgmf_kernel_size) * 2 - 1), 4., mode="constant", cval=0))
        self.dx = nn.Parameter(dx, requires_grad=True)
        self.dy = nn.Parameter(dy, requires_grad=True)

        self.alphas.data.uniform_(-5, 5)
        self.dgmf = ApplyDGMF(self.dgmf_kernel_size, sigmas=self.sigmas, alphas=self.alphas, dx=self.dx, dy=self.dy,
                              order=self.dgmf_order)

        self.thin = self.base_model(n_channels + n_dgmf // 2, self.n_classes)
        self.thick = self.base_model(n_channels + n_dgmf // 2, self.n_classes)
        #
        self.att1 = self.att_layer(self.n_dgmf, self.n_dgmf)
        self.att2 = self.att_layer(self.n_classes * 16, self.n_classes * 2)
        self.final = nn.Sequential(nn.Conv2d(self.n_classes, self.n_classes, 1), )

    def forward(self, x):
        self.dgmf = ApplyDGMF(self.dgmf_kernel_size, sigmas=self.sigmas,
                              alphas=self.alphas, dx=self.dx, dy=self.dy,
                              order=self.dgmf_order)
        dgmf1 = self.dgmf(1. - x)
        drop1 = self.dropout2d(dgmf1)

        att_out1 = self.att1(drop1)
        dgmf_att = drop1 * att_out1
        cat_input_thin = torch.cat([x, dgmf_att[:, :self.n_dgmf // 2]], dim=1)
        cat_input_thick = torch.cat([x, dgmf_att[:, self.n_dgmf // 2:]], dim=1)
        out_thin = self.thin(cat_input_thin)
        out_thick = self.thick(cat_input_thick)

        cat_outs = torch.cat([out_thin, out_thick] * 8, dim=1)

        if self.att2 is not None:
            att_outs = self.att2(cat_outs)
            out_thin = out_thin * att_outs[:, :self.n_classes]
            out_thick = out_thick * att_outs[:, self.n_classes:]
        out = out_thin + out_thick
        out = self.final(out)

        return out_thin, out_thick, out

"""
#how to use:

torch.manual_seed(0)
# filter size
n = 7
# input in format BxCxHxW
dummy = torch.rand(1, 3, 48, 48)
b, c, h, w = dummy.shape
# create model
model = WANet(in_channels=3, n_classes=2, dgmf_kernel_size=(n,n))
# apply model
output_thin, output_thick, output = model(dummy)  # 1x3x15x15
# first output
print(output.shape)
"""