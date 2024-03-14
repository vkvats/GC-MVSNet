import torch
import torch.nn as nn
import torch.nn.functional as F
from .dcn import DCN


def init_bn(module):
    if module.weight is not None:
        nn.init.ones_(module.weight)
    if module.bias is not None:
        nn.init.zeros_(module.bias)
    return


def init_uniform(module, init_method):
    if module.weight is not None:
        if init_method == "kaiming":
            nn.init.kaiming_uniform_(module.weight, mode='fan_in', nonlinearity='relu')
        elif init_method == "xavier":
            nn.init.xavier_uniform_(module.weight)
    return


## ---------------------------------------Kernel standardization method: ---------------------------------------
## paper: https://arxiv.org/abs/1903.10520
class WeightStandardizedConv2d(nn.Conv2d):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=False, **kwargs):
        super(WeightStandardizedConv2d, self).__init__(in_channels, out_channels, kernel_size, stride,
                 padding, dilation, groups, bias, **kwargs)

    def forward(self, x):
        weight = self.weight
        weight_mean = weight.mean(dim=1, keepdim=True).mean(dim=2,
                                  keepdim=True).mean(dim=3, keepdim=True)
        weight = weight - weight_mean
        std = weight.view(weight.size(0), -1).std(dim=1).view(-1, 1, 1, 1) + 1e-5
        weight = weight / std.expand_as(weight)
        return F.conv2d(x, weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)

## Kernel standardization method for 3DConv layers, based on kernel standardization method
## paper: https://arxiv.org/abs/1903.10520 
class WeightStandardizedConv3d(nn.Conv3d):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, **kwargs):
        super(WeightStandardizedConv3d, self).__init__(in_channels, out_channels, kernel_size, stride,
                 padding, dilation, groups, bias, **kwargs)

    def forward(self, x):
        weight = self.weight
        weight_mean = weight.mean(dim=1, keepdim=True).mean(dim=2,
                                  keepdim=True).mean(dim=3, keepdim=True).mean(dim=4, keepdim=True)
        weight = weight - weight_mean
        std = weight.view(weight.size(0), -1).std(dim=1).view(-1, 1, 1, 1, 1) + 1e-5
        weight = weight / std.expand_as(weight)
        return F.conv3d(x, weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)

## Kernel standardization method for 3DConv layers, based on kernel standardization method
## paper: https://arxiv.org/abs/1903.10520 
class WeightStandardizedConvTrans3d(nn.ConvTranspose3d):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, output_padding=1, dilation=1, groups=1, bias=True, **kwargs):
        super(WeightStandardizedConvTrans3d, self).__init__(in_channels, out_channels, kernel_size, stride,
                 padding, output_padding, groups, bias, dilation, **kwargs)

    def forward(self, x):
        weight = self.weight
        # print(weight.shape)
        weight_mean = weight.mean(dim=1, keepdim=True).mean(dim=2,
                                  keepdim=True).mean(dim=3, keepdim=True).mean(dim=4, keepdim=True)
        weight = weight - weight_mean
        std = weight.view(weight.size(0), -1).std(dim=1).view(-1, 1, 1, 1, 1) + 1e-5
        weight = weight / std.expand_as(weight)
        return F.conv_transpose3d(x, weight, self.bias, self.stride,
                                padding=self.padding, 
                                output_padding=self.output_padding, 
                                dilation=self.dilation, 
                                groups=self.groups)

##-----------------------------------------------------------------------------------------------------------------


class Conv2d(nn.Module):
    """Applies a 2D convolution (optionally with batch normalization and relu activation)
    over an input signal composed of several input planes.

    Attributes:
        conv (nn.Module): convolution module
        bn (nn.Module): batch normalization module
        relu (bool): whether to activate by relu

    Notes:
        Default momentum for batch normalization is set to be 0.01,

    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 relu=True, gn_group=4, bn_momentum=0.1, init_method="kaiming", **kwargs):
        super(Conv2d, self).__init__()

        self.conv = WeightStandardizedConv2d(in_channels, out_channels, kernel_size, stride=stride, 
                              **kwargs)
        self.kernel_size = kernel_size
        self.stride = stride
        self.gn = nn.GroupNorm(num_groups=gn_group, 
                                num_channels=out_channels, 
                                eps=1e-05, affine=True)
        self.relu = relu
        self.init_weights(init_method)

    def forward(self, x):
        x = self.conv(x)
        x = self.gn(x)
        if self.relu:
            x = F.relu(x, inplace=True)
        return x

    def init_weights(self, init_method):
        """default initialization"""
        init_uniform(self.conv, init_method)


class DenseTransitionConv2D(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=2,
                 relu=True, gn_group=4, bn_momentum=0.1, init_method="kaiming", **kwargs):
        super(DenseTransitionConv2D, self).__init__()

        self.conv = WeightStandardizedConv2d(in_channels, out_channels, kernel_size, stride=stride,
                              bias=False, **kwargs)
        self.kernel_size = kernel_size
        self.stride = stride
        self.gn = nn.GroupNorm(num_groups=gn_group, 
                                num_channels=out_channels, 
                                eps=1e-05, affine=True)
        self.relu = relu
        self.init_weights(init_method)

    def forward(self, x):
        ## strided Conv Branch
        x1 = self.conv(x)
        x1 = self.gn(x1)
        if self.relu:
            x1 = F.relu(x1, inplace=True)
        ## interpolation branch
        x = F.interpolate(x, scale_factor=0.5, mode="nearest")
        return torch.cat([x1,x], dim=1)

    def init_weights(self, init_method):
        """default initialization"""
        init_uniform(self.conv, init_method)



class Deconv2d(nn.Module):
    """Applies a 2D deconvolution (optionally with batch normalization and relu activation)
       over an input signal composed of several input planes.

       Attributes:
           conv (nn.Module): convolution module
           bn (nn.Module): batch normalization module
           relu (bool): whether to activate by relu

       Notes:
           Default momentum for batch normalization is set to be 0.01,

       """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 relu=True, bn=True, bn_momentum=0.1, init_method="kaiming", **kwargs):
        super(Deconv2d, self).__init__()
        self.out_channels = out_channels
        assert stride in [1, 2]
        self.stride = stride

        self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride,
                                       bias=False, **kwargs)
        self.gn = nn.GroupNorm(num_groups=gn_group, 
                                num_channels=out_channels, 
                                eps=1e-05, affine=True)
        self.relu = relu

    def forward(self, x):
        y = self.conv(x)
        if self.stride == 2:
            h, w = list(x.size())[2:]
            y = y[:, :, :2 * h, :2 * w].contiguous()
        x = self.gn(y)
        if self.relu:
            x = F.relu(x, inplace=True)
        return x

    def init_weights(self, init_method):
        """default initialization"""
        init_uniform(self.conv, init_method)


class Conv3d(nn.Module):
    """Applies a 3D convolution (optionally with batch normalization and relu activation)
    over an input signal composed of several input planes.

    Attributes:
        conv (nn.Module): convolution module
        bn (nn.Module): batch normalization module
        relu (bool): whether to activate by relu

    Notes:
        Default momentum for batch normalization is set to be 0.01,

    """

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 relu=True, gn_group=4, bn_momentum=0.1, init_method="kaiming", **kwargs):
        super(Conv3d, self).__init__()
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        assert stride in [1, 2]
        self.stride = stride

        self.conv = WeightStandardizedConv3d(in_channels, out_channels, kernel_size, stride=stride,
                              bias=False, **kwargs)
        self.gn = nn.GroupNorm(num_groups=gn_group, 
                                num_channels=out_channels, 
                                eps=1e-05, affine=True)
        self.relu = relu
        self.init_weights(init_method)

    def forward(self, x):
        x = self.conv(x)
        x = self.gn(x)
        if self.relu:
            x = F.relu(x, inplace=True)
        return x

    def init_weights(self, init_method):
        """default initialization"""
        init_uniform(self.conv, init_method)


class Deconv3d(nn.Module):
    """Applies a 3D deconvolution (optionally with batch normalization and relu activation)
       over an input signal composed of several input planes.

       Attributes:
           conv (nn.Module): convolution module
           bn (nn.Module): batch normalization module
           relu (bool): whether to activate by relu

       Notes:
           Default momentum for batch normalization is set to be 0.01,

       """

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 relu=True, gn_group=4, bn_momentum=0.1, init_method="kaiming", **kwargs):
        super(Deconv3d, self).__init__()
        self.out_channels = out_channels
        assert stride in [1, 2]
        self.stride = stride

        self.conv = WeightStandardizedConvTrans3d(in_channels, out_channels, kernel_size, stride=stride,
                                       bias=False, **kwargs)
        self.gn = nn.GroupNorm(num_groups=gn_group, 
                                num_channels=out_channels, 
                                eps=1e-05, affine=True)
        self.relu = relu

        # assert init_method in ["kaiming", "xavier"]
        self.init_weights(init_method)

    def forward(self, x):
        y = self.conv(x)
        x = self.gn(y)
        if self.relu:
            x = F.relu(x, inplace=True)
        return x

    def init_weights(self, init_method):
        """default initialization"""
        init_uniform(self.conv, init_method)


class ConvBnReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, pad=1, gn_group=4):
        super(ConvBnReLU, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=pad, bias=False)
        self.gn = nn.GroupNorm(num_groups=gn_group, 
                                num_channels=out_channels, 
                                eps=1e-05, affine=True)

    def forward(self, x):
        return F.relu(self.gn(self.conv(x)), inplace=True)


class ConvBn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, pad=1, gn_group=4):
        super(ConvBn, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=pad, bias=False)
        self.gn = nn.GroupNorm(num_groups=gn_group, 
                                num_channels=out_channels, 
                                eps=1e-05, affine=True)

    def forward(self, x):
        return self.gn(self.conv(x))


class ConvBnReLU3D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, pad=1, gn_group=4):
        super(ConvBnReLU3D, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=pad, bias=False)
        self.gn = nn.GroupNorm(num_groups=gn_group, 
                                num_channels=out_channels, 
                                eps=1e-05, affine=True)

    def forward(self, x):
        return F.relu(self.gn(self.conv(x)), inplace=True)


class ConvBn3D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, pad=1, gn_group=4):
        super(ConvBn3D, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=pad, bias=False)
        self.gn = nn.GroupNorm(num_groups=gn_group, 
                                num_channels=out_channels, 
                                eps=1e-05, affine=True)

    def forward(self, x):
        return self.gn(self.conv(x))


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, downsample=None):
        super(BasicBlock, self).__init__()

        self.conv1 = ConvBnReLU(in_channels, out_channels, kernel_size=3, stride=stride, pad=1)
        self.conv2 = ConvBn(out_channels, out_channels, kernel_size=3, stride=1, pad=1)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        if self.downsample is not None:
            x = self.downsample(x)
        out += x
        return out


def homo_warping(src_fea, src_proj, ref_proj, depth_values):
    # src_fea: [B, C, H, W]
    # src_proj: [B, 4, 4]
    # ref_proj: [B, 4, 4]
    # depth_values: [B, Ndepth] or [B, Ndepth, H, W]
    # out: [B, C, Ndepth, H, W]
    batch, channels = src_fea.shape[0], src_fea.shape[1]
    num_depth = depth_values.shape[1]
    height, width = src_fea.shape[2], src_fea.shape[3]

    with torch.no_grad():
        proj = torch.matmul(src_proj, torch.inverse(ref_proj))
        rot = proj[:, :3, :3]  # [B,3,3]
        trans = proj[:, :3, 3:4]  # [B,3,1]

        y, x = torch.meshgrid([torch.arange(0, height, dtype=torch.float32, device=src_fea.device),
                               torch.arange(0, width, dtype=torch.float32, device=src_fea.device)])
        y, x = y.contiguous(), x.contiguous()
        y, x = y.view(height * width), x.view(height * width)
        xyz = torch.stack((x, y, torch.ones_like(x)))  # [3, H*W]
        xyz = torch.unsqueeze(xyz, 0).repeat(batch, 1, 1)  # [B, 3, H*W]
        rot_xyz = torch.matmul(rot, xyz)  # [B, 3, H*W]
        rot_depth_xyz = rot_xyz.unsqueeze(2).repeat(1, 1, num_depth, 1) * depth_values.view(batch, 1, num_depth,
                                                                                            -1)  # [B, 3, Ndepth, H*W]
        proj_xyz = rot_depth_xyz + trans.view(batch, 3, 1, 1)  # [B, 3, Ndepth, H*W]
        invalid = (proj_xyz[:, 2:3, :, :]<1e-6).squeeze(1) # [B, Ndepth, H*W]
        proj_xy = proj_xyz[:, :2, :, :] / (proj_xyz[:, 2:3, :, :])  # [B, 2, Ndepth, H*W]
        proj_x_normalized = proj_xy[:, 0, :, :] / ((width - 1) / 2) - 1
        proj_x_normalized[invalid] = -99.
        proj_y_normalized = proj_xy[:, 1, :, :] / ((height - 1) / 2) - 1
        proj_y_normalized[invalid] = -99.
        proj_xy = torch.stack((proj_x_normalized, proj_y_normalized), dim=3)  # [B, Ndepth, H*W, 2]
        grid = proj_xy

    warped_src_fea = F.grid_sample(src_fea, grid.view(batch, num_depth * height, width, 2), mode='bilinear',
                                   padding_mode='zeros', align_corners=True)
    warped_src_fea = warped_src_fea.view(batch, channels, num_depth, height, width)

    return warped_src_fea


class DeConv2dFuse(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, relu=True, bn=True,
                 bn_momentum=0.1):
        super(DeConv2dFuse, self).__init__()

        self.deconv = Deconv2d(in_channels, out_channels, kernel_size, stride=2, padding=1, output_padding=1,
                               bn=True, relu=relu, bn_momentum=bn_momentum)

        self.conv = Conv2d(2*out_channels, out_channels, kernel_size, stride=1, padding=1,
                           bn=bn, relu=relu, bn_momentum=bn_momentum)

    def forward(self, x_pre, x):
        x = self.deconv(x)
        x = torch.cat((x, x_pre), dim=1)
        x = self.conv(x)
        return x


def get_cur_depth_range_samples(cur_depth, ndepth, depth_inteval_pixel, shape, max_depth=192.0, min_depth=0.0):
    cur_depth_min = (cur_depth - ndepth / 2 * depth_inteval_pixel)  # (B, H, W)
    cur_depth_max = (cur_depth + ndepth / 2 * depth_inteval_pixel)
    assert cur_depth.shape == torch.Size(shape), "cur_depth:{}, input shape:{}".format(cur_depth.shape, shape)
    new_interval = (cur_depth_max - cur_depth_min) / (ndepth - 1)  # (B, H, W)
    
    depth_range_samples = cur_depth_min.unsqueeze(1) + (torch.arange(0, ndepth, device=cur_depth.device,
                                                                  dtype=cur_depth.dtype,
                                                                  requires_grad=False).reshape(1, -1, 1, 1) * new_interval.unsqueeze(1))

    return depth_range_samples


def get_depth_range_samples(cur_depth, ndepth, depth_inteval_pixel, device, dtype, shape,
                           max_depth=192.0, min_depth=0.0, use_inverse_depth=False):
    # print(cur_depth.dim())
    if cur_depth.dim() == 2:
        cur_depth_min = cur_depth[:, 0]  # (B,)
        cur_depth_max = cur_depth[:, -1]
        
        if use_inverse_depth is False:
            new_interval = (cur_depth_max - cur_depth_min) / (ndepth - 1)  # (B, )  Shouldn't cal this if we use inverse depth
            
            depth_range_samples = cur_depth_min.unsqueeze(1) + (torch.arange(0, ndepth, device=device, dtype=dtype,
                                                                        requires_grad=False).reshape(1, -1) * new_interval.unsqueeze(1)) #(B, D)
            depth_range_samples = depth_range_samples.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, shape[1], shape[2]) #(B, D, H, W)
        else:
            # When use inverse_depth for T&T
            depth_range_samples = cur_depth.repeat(1, 1, shape[1], shape[2]) #(B, D, H, W)

    else:
        depth_range_samples = get_cur_depth_range_samples(cur_depth, ndepth, depth_inteval_pixel, shape, max_depth, min_depth)

    return depth_range_samples

