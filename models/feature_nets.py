import torch
import torch.nn as nn
import torch.nn.functional as F
from .dcn import *
from .module import *


## ---------------------- TransMVSNet feature net --------------------------------------
class FeatureNet(nn.Module):
    def __init__(self, base_channels):
        super(FeatureNet, self).__init__()
        self.base_channels = base_channels

        self.conv0 = nn.Sequential(
                Conv2d(3, base_channels, 3, 1, padding=1),
                Conv2d(base_channels, base_channels, 3, 1, padding=1))

        self.conv1 = nn.Sequential(
                Conv2d(base_channels, base_channels * 2, 5, stride=2, padding=2),
                Conv2d(base_channels * 2, base_channels * 2, 3, 1, padding=1),
                Conv2d(base_channels * 2, base_channels * 2, 3, 1, padding=1))

        self.conv2 = nn.Sequential(
                Conv2d(base_channels * 2, base_channels * 4, 5, stride=2, padding=2),
                Conv2d(base_channels * 4, base_channels * 4, 3, 1, padding=1),
                Conv2d(base_channels * 4, base_channels * 4, 3, 1, padding=1))

        self.out1 = nn.Sequential(
                Conv2d(base_channels * 4, base_channels * 4, 1),
                DCN(in_channels=base_channels * 4, out_channels=base_channels * 4, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(base_channels * 4),
                nn.ReLU(inplace=True),
                DCN(in_channels=base_channels * 4, out_channels=base_channels * 4, kernel_size=3,stride=1, padding=1),
                nn.BatchNorm2d(base_channels * 4),
                nn.ReLU(inplace=True),
                DCN(in_channels=base_channels * 4, out_channels=base_channels * 4, kernel_size=3,stride=1, padding=1))


        final_chs = base_channels * 4
        self.inner1 = nn.Conv2d(base_channels * 2, final_chs, 1, bias=True)
        self.inner2 = nn.Conv2d(base_channels * 1, final_chs, 1, bias=True)

        self.out2 = nn.Sequential(
                Conv2d(final_chs, final_chs, 3,1,padding=1),
                DCN(in_channels=final_chs, out_channels=final_chs,kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(final_chs),
                nn.ReLU(inplace=True),
                DCN(in_channels=final_chs, out_channels=final_chs,kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(final_chs),
                nn.ReLU(inplace=True),
                DCN(in_channels=final_chs, out_channels=base_channels * 2,kernel_size=3, stride=1, padding=1),
                                  )
        self.out3 = nn.Sequential(
                Conv2d(final_chs, final_chs, 3, 1, padding=1),
                DCN(in_channels=final_chs, out_channels=final_chs, kernel_size=3, stride=1,padding=1),
                nn.BatchNorm2d(final_chs),
                nn.ReLU(inplace=True),
                DCN(in_channels=final_chs, out_channels=final_chs, kernel_size=3, stride=1,padding=1),
                nn.BatchNorm2d(final_chs),
                nn.ReLU(inplace=True),
                DCN(in_channels=final_chs, out_channels=base_channels, kernel_size=3,stride=1, padding=1))

        self.out_channels = [4 * base_channels, base_channels * 2, base_channels]

    def forward(self, x):
        """forward.

        :param x: [B, C, H, W]
        :return outputs: stage1 [B, 32， 128， 160], stage2 [B, 16, 256, 320], stage3 [B, 8, 512, 640]
        """
        conv0 = self.conv0(x)
        conv1 = self.conv1(conv0)
        conv2 = self.conv2(conv1)

        intra_feat = conv2
        outputs = {}
        out = self.out1(intra_feat)
        outputs["stage1"] = out
        intra_feat = F.interpolate(intra_feat, scale_factor=2, mode="nearest") + self.inner1(conv1)
        out = self.out2(intra_feat)
        outputs["stage2"] = out

        intra_feat = F.interpolate(intra_feat, scale_factor=2, mode="nearest") + self.inner2(conv0)
        out = self.out3(intra_feat)
        outputs["stage3"] = out

        return outputs

#### ----------------------- our feature net integrated with TransMVS feature net -----------------------
class DeformableFeatureNet(nn.Module):
    def __init__(self, base_channels):
        super(DeformableFeatureNet, self).__init__()
        self.base_channels = base_channels
        self.gn_g = 4

        self.conv0 = nn.Sequential(
            DeformableConv2d(3, base_channels, 3, 1, padding=1, gn_group=self.gn_g),
            DeformableConv2d(base_channels, base_channels, 3, 1, padding=1, gn_group=self.gn_g),
        )

        self.conv1 = nn.Sequential(
            DeformableConv2d(base_channels, base_channels * 2, 5, stride=2, padding=2, gn_group=self.gn_g),
            DeformableConv2d(base_channels * 2, base_channels * 2, 3, 1, padding=1, gn_group=self.gn_g),
            DeformableConv2d(base_channels * 2, base_channels * 2, 3, 1, padding=1, gn_group=self.gn_g),
        )

        self.conv2 = nn.Sequential(
            DeformableConv2d(base_channels * 2, base_channels * 4, 5, stride=2, padding=2, gn_group=self.gn_g),
            DeformableConv2d(base_channels * 4, base_channels * 4, 3, 1, padding=1, gn_group=self.gn_g),
            DeformableConv2d(base_channels * 4, base_channels * 4, 3, 1, padding=1, gn_group=self.gn_g),
        )

        self.out1 = nn.Sequential(
                Conv2d(base_channels * 4, base_channels * 4, 1),
                DCN(in_channels=base_channels * 4, out_channels=base_channels * 4, kernel_size=3, stride=1, padding=1),
                nn.GroupNorm(num_groups=self.gn_g, num_channels=base_channels*4, eps=1e-05, affine=True),
                nn.ReLU(inplace=True),
                DCN(in_channels=base_channels * 4, out_channels=base_channels * 4, kernel_size=3,stride=1, padding=1),
                nn.GroupNorm(num_groups=self.gn_g, num_channels=base_channels*4, eps=1e-05, affine=True),
                nn.ReLU(inplace=True),
                DCN(in_channels=base_channels * 4, out_channels=base_channels * 4, kernel_size=3,stride=1, padding=1))


        final_chs = base_channels * 4
        self.inner1 = nn.Conv2d(base_channels * 2, final_chs, 1, bias=True)
        self.inner2 = nn.Conv2d(base_channels * 1, final_chs, 1, bias=True)

        self.out2 = nn.Sequential(
                DeformableConv2d(final_chs, final_chs, 3, 1, padding=1, gn_group=self.gn_g),
                DCN(in_channels=final_chs, out_channels=final_chs,kernel_size=3, stride=1, padding=1),
                nn.GroupNorm(num_groups=self.gn_g, num_channels=final_chs, eps=1e-05, affine=True),
                nn.ReLU(inplace=True),
                DCN(in_channels=final_chs, out_channels=final_chs,kernel_size=3, stride=1, padding=1),
                nn.GroupNorm(num_groups=self.gn_g, num_channels=final_chs, eps=1e-05, affine=True),
                nn.ReLU(inplace=True),
                DCN(in_channels=final_chs, out_channels=base_channels * 2,kernel_size=3, stride=1, padding=1),
                                  )
        self.out3 = nn.Sequential(
                DeformableConv2d(final_chs, final_chs, 3, 1, padding=1, gn_group=self.gn_g),
                DCN(in_channels=final_chs, out_channels=final_chs, kernel_size=3, stride=1,padding=1),
                nn.GroupNorm(num_groups=self.gn_g, num_channels=final_chs, eps=1e-05, affine=True),
                nn.ReLU(inplace=True),
                DCN(in_channels=final_chs, out_channels=final_chs, kernel_size=3, stride=1,padding=1),
                nn.GroupNorm(num_groups=self.gn_g, num_channels=final_chs, eps=1e-05, affine=True),
                nn.ReLU(inplace=True),
                DCN(in_channels=final_chs, out_channels=base_channels, kernel_size=3,stride=1, padding=1))

        self.out_channels = [4 * base_channels, base_channels * 2, base_channels]

    def forward(self, x):
        """forward.

        :param x: [B, C, H, W]
        :return outputs: stage1 [B, 32， 128， 160], stage2 [B, 16, 256, 320], stage3 [B, 8, 512, 640]
        """
        conv0 = self.conv0(x)
        conv1 = self.conv1(conv0)
        conv2 = self.conv2(conv1)

        intra_feat = conv2
        outputs = {}
        out = self.out1(intra_feat)
        outputs["stage1"] = out
        intra_feat = F.interpolate(intra_feat, scale_factor=2, mode="nearest") + self.inner1(conv1)
        out = self.out2(intra_feat)
        outputs["stage2"] = out

        intra_feat = F.interpolate(intra_feat, scale_factor=2, mode="nearest") + self.inner2(conv0)
        out = self.out3(intra_feat)
        outputs["stage3"] = out

        return outputs


