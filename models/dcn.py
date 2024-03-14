#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import math
import torch
from torch import nn
from torch.autograd import Function
from torch.nn.modules.utils import _pair
from torch.autograd.function import once_differentiable
from torchvision.ops import DeformConv2d, deform_conv2d

## imports
import torchvision.ops
import torch.nn.functional as F


class DCNv2(nn.Module):

    def __init__(self, in_channels, out_channels,
                 kernel_size, stride, padding, dilation=1, deformable_groups=1):
        super(DCNv2, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.deformable_groups = deformable_groups

        self.weight = nn.Parameter(torch.Tensor(
            out_channels, in_channels, *self.kernel_size))
        self.bias = nn.Parameter(torch.Tensor(out_channels))
        self.reset_parameters()

    def reset_parameters(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1. / math.sqrt(n)
        self.weight.data.uniform_(-stdv, stdv)
        self.bias.data.zero_()



class DCN(DCNv2):

    def __init__(self, in_channels, out_channels,
                 kernel_size, stride, padding,
                 dilation=1, deformable_groups=1,bias=True):
        super(DCN, self).__init__(in_channels, out_channels,
                                  kernel_size, stride, padding, dilation, deformable_groups)

        channels_ = self.deformable_groups * 3 * self.kernel_size[0] * self.kernel_size[1]
        self.conv_offset_mask = nn.Conv2d(self.in_channels,
                                          channels_,
                                          kernel_size=self.kernel_size,
                                          stride=self.stride,
                                          padding=self.padding,
                                          bias=True)
        if bias==False:
            self.bias = None
        self.init_offset()

    def init_offset(self):
        self.conv_offset_mask.weight.data.zero_()
        self.conv_offset_mask.bias.data.zero_()

    def forward(self, input):
        out = self.conv_offset_mask(input)
        o1, o2, mask = torch.chunk(out, 3, dim=1)
        offset = torch.cat((o1, o2), dim=1)
        mask = torch.sigmoid(mask)
        return deform_conv2d(
            input,
            offset,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            mask=mask
        )
############################################### DCN from our experiments ###########################################
## My version of Deconvolution Network: 

## common functions 
weight_init_method = "he"

def init_bn(module):
    if module.weight is not None:
        nn.init.ones_(module.weight)
    if module.bias is not None:
        nn.init.zeros_(module.bias)
    return


def init_uniform(module, init_method):
    if module.weight is not None:
        if init_method == "he":
            nn.init.kaiming_uniform_(module.weight, mode='fan_in', nonlinearity='relu')
        elif init_method == "xavier":
            nn.init.xavier_uniform_(module.weight)
        elif init_method == "truc_norm":
            nn.init.trunc_normal_(module.weight)
    return

### Kernel standardization method: paper: https://arxiv.org/abs/1903.10520
class WeightStandardizedConv2d(nn.Conv2d):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, **kwargs):
        super(WeightStandardizedConv2d, self).__init__(in_channels, out_channels, kernel_size, stride,
                 padding, dilation, groups, bias, **kwargs)

    def forward(self, x):
        # return super(Conv2d, self).forward(x)
        weight = self.weight
        weight_mean = weight.mean(dim=1, keepdim=True).mean(dim=2,
                                  keepdim=True).mean(dim=3, keepdim=True)
        weight = weight - weight_mean
        std = weight.view(weight.size(0), -1).std(dim=1).view(-1, 1, 1, 1) + 1e-5
        weight = weight / std.expand_as(weight)
        return F.conv2d(x, weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)


class DeformableConv2dLayer(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 padding=1,
                 bias=False,
                 init_method=weight_init_method,
                 **kwargs):

        super(DeformableConv2dLayer, self).__init__()
        
        assert type(kernel_size) == tuple or type(kernel_size) == int

        kernel_size = kernel_size if type(kernel_size) == tuple else (kernel_size, kernel_size)
        self.stride = stride if type(stride) == tuple else (stride, stride)
        self.padding = padding
        
        self.offset_conv = nn.Conv2d(in_channels, 
                                     2 * kernel_size[0] * kernel_size[1],
                                     kernel_size=kernel_size, 
                                     stride=stride,
                                     padding=self.padding, 
                                     bias=True)

        nn.init.constant_(self.offset_conv.weight, 0.)
        nn.init.constant_(self.offset_conv.bias, 0.)
        
        self.modulator_conv = nn.Conv2d(in_channels, 
                                     1 * kernel_size[0] * kernel_size[1],
                                     kernel_size=kernel_size, 
                                     stride=stride,
                                     padding=self.padding, 
                                     bias=True)

        nn.init.constant_(self.modulator_conv.weight, 0.)
        nn.init.constant_(self.modulator_conv.bias, 0.)
        
        self.regular_conv = nn.Conv2d(in_channels=in_channels,
                                      out_channels=out_channels,
                                      kernel_size=kernel_size,
                                      stride=stride,
                                      padding=self.padding,
                                      bias=bias)
        ## init weights
        self.init_weights(init_method)
    
    def init_weights(self, init_method):
        """default initialization"""
        init_uniform(self.regular_conv, init_method)

    def forward(self, x):

        offset = self.offset_conv(x)
        modulator = 2. * torch.sigmoid(self.modulator_conv(x))
        #  Notes: offset and mask should have same size as the output of the convolution 
        # 
        
        x = torchvision.ops.deform_conv2d(input=x, 
                                          offset=offset, 
                                          weight=self.regular_conv.weight, 
                                          bias=self.regular_conv.bias, 
                                          padding=self.padding,
                                          mask=modulator,
                                          stride=self.stride,
                                          )
        return x
    

class DeformableConv2d(nn.Module):
    """Applies a deformable 2D convolution (optionally with batch normalization and relu activation)
    over an input signal composed of several input planes.

    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 relu=True, gn_group=4, bn_momentum=0.1, **kwargs):
        super(DeformableConv2d, self).__init__()
       
        self.conv = DeformableConv2dLayer(in_channels=in_channels, out_channels=out_channels, 
                                          kernel_size=kernel_size, stride=stride, bias=False, **kwargs)
        
        self.kernel_size = kernel_size
        self.stride = stride
        self.gn = nn.GroupNorm(num_groups=gn_group, 
                                num_channels=out_channels, 
                                eps=1e-05, affine=True)
        self.relu = relu
        

    def forward(self, x):
        x = self.conv(x)
        x = self.gn(x)
        if self.relu:
            x = F.relu(x, inplace=True)
        return x


class DeformableTransitionConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 relu=True, gn_group=4, bn_momentum=0.1, **kwargs):
        super(DeformableTransitionConv2d, self).__init__()
       
        self.conv = DeformableConv2dLayer(in_channels=in_channels, out_channels=out_channels, 
                                          kernel_size=kernel_size, stride=stride, bias=False, **kwargs)
        
        self.kernel_size = kernel_size
        self.stride = stride
        self.gn = nn.GroupNorm(num_groups=gn_group, 
                                num_channels=out_channels, 
                                eps=1e-05, affine=True)
        self.relu = relu
        

    def forward(self, x):
        ## strided Conv branch
        x1 = self.conv(x)
        x1 = self.gn(x1)
        if self.relu:
            x1 = F.relu(x1, inplace=True)
        ## interpolation branch
        x = F.interpolate(x, scale_factor=0.5, mode="nearest")
        return torch.cat([x1,x], dim=1)