import torch
import torch.nn as nn

########################################################################################################################
# UNet
########################################################################################################################

def double_conv(in_channels, out_channels, p_dropout=0.2, padding=1,
                batchnorm=True, activation=True):
    layers = [nn.Conv2d(in_channels, out_channels, 3, padding=padding, groups=1)]
    if batchnorm:
        layers.append(nn.BatchNorm2d(out_channels))
    if activation:
        layers.append(nn.ReLU(inplace=False))
    layers.append(nn.Dropout2d(p=p_dropout))
    layers.append(nn.Conv2d(out_channels, out_channels, 3, padding=padding, groups=1))
    if batchnorm:
        layers.append(nn.BatchNorm2d(out_channels))
    if activation:
        layers.append(nn.ReLU(inplace=False))

    return nn.Sequential(*layers)

class UNet(nn.Module):
    def __init__(self, in_channels, n_classes=2, seg_grad_cam=False):
        super().__init__()
        self.n_classes = n_classes
        self.seg_grad_cam = seg_grad_cam
        self.down1 = double_conv(in_channels, 32)
        self.down2 = double_conv(32, 64)
        self.down3 = double_conv(64, 128)
        self.middle = double_conv(128, 256)
        self.up1 = double_conv(256+128, 128)
        self.up2 = double_conv(128+64, 64)
        self.up3 = double_conv(64+32, 32)

        self.maxpool = nn.MaxPool2d(2, stride=2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.final = nn.Sequential(nn.Conv2d(32, self.n_classes, 1),)#nn.Conv2d(32, self.n_classes, 1)
        self.conv1 = None
        self.conv2 = None
        self.conv3 = None

    def forward(self,  x):
        self.conv1 = self.down1(x)
        x = self.maxpool(self.conv1)
        self.conv2 = self.down2(x)
        x = self.maxpool(self.conv2)
        self.conv3 = self.down3(x)
        x = self.maxpool(self.conv3)

        x = self.middle(x)

        x = self.upsample(x)
        x = torch.cat([x, self.conv3], dim=1)
        x = self.up1(x)

        x = self.upsample(x)
        x = torch.cat([x, self.conv2], dim=1)
        x = self.up2(x)

        x = self.upsample(x)
        x = torch.cat([x, self.conv1], dim=1)
        x = self.up3(x)

        if not self.seg_grad_cam:
            x = self.final(x)

        return x
