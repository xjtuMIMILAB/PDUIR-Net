import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
# from timm.models.layers import trunc_normal_

def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x

class Window_Hybrid_Attention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        channel_scale: number of branches as input.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, channel_scale=1, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0., norm_layer=nn.LayerNorm):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = self.dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.channel_scale = channel_scale
        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv1 = nn.Linear(self.dim, self.dim * 3 // self.channel_scale, bias=qkv_bias)
        self.qkv2 = nn.Linear(self.dim, self.dim * 3 // self.channel_scale, bias=qkv_bias)
        self.qkv3 = nn.Linear(self.dim, self.dim * 3 // self.channel_scale, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(self.dim//self.channel_scale, dim//self.channel_scale)

        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

        self.norm_layer = norm_layer
        self.norm = self.norm_layer(self.dim)

    def cal_WHA(self, x, y, z, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        x_qkv = self.qkv1(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads//self.channel_scale).permute(2, 0, 3, 4, 1) # (2, 0, 3, 1, 4)
        y_qkv = self.qkv2(y).reshape(B_, N, 3, self.num_heads, C // self.num_heads//self.channel_scale).permute(2, 0, 3, 4, 1)
        z_qkv = self.qkv3(z).reshape(B_, N, 3, self.num_heads, C // self.num_heads//self.channel_scale).permute(2, 0, 3, 4, 1)
        x_q, x_k, x_v = x_qkv[0], x_qkv[1], x_qkv[2]  # make torchscript happy (cannot use tensor as tuple)
        y_q, y_k, y_v = y_qkv[0], y_qkv[1], y_qkv[2]
        z_q, z_k, z_v = z_qkv[0], z_qkv[1], z_qkv[2]
        x_q = x_q * self.scale
        y_q = y_q * self.scale
        z_q = z_q * self.scale
        x2x_attn = (x_q @ x_k.transpose(-2, -1)); x2y_attn = (y_q @ x_k.transpose(-2, -1)); x2z_attn = (z_q @ x_k.transpose(-2, -1))
        y2x_attn = (x_q @ y_k.transpose(-2, -1)); y2y_attn = (y_q @ y_k.transpose(-2, -1)); y2z_attn = (z_q @ y_k.transpose(-2, -1))
        z2x_attn = (x_q @ z_k.transpose(-2, -1)); z2y_attn = (y_q @ z_k.transpose(-2, -1)); z2z_attn = (z_q @ z_k.transpose(-2, -1))

        # relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
        #     self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        # relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        # x2x_attn = x2x_attn + relative_position_bias.unsqueeze(0); x2y_attn = x2y_attn + relative_position_bias.unsqueeze(0); x2z_attn = x2z_attn + relative_position_bias.unsqueeze(0)
        # y2x_attn = y2x_attn + relative_position_bias.unsqueeze(0); y2y_attn = y2y_attn + relative_position_bias.unsqueeze(0); y2z_attn = y2z_attn + relative_position_bias.unsqueeze(0)
        # z2x_attn = z2x_attn + relative_position_bias.unsqueeze(0); z2y_attn = z2y_attn + relative_position_bias.unsqueeze(0); z2z_attn = z2z_attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            # nW = mask.shape[0]
            # attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            # attn = attn.view(-1, self.num_heads, N, N)
            # attn = self.softmax(attn)
            print("Not support mask !!!")
        else:
            x2x_attn = self.softmax(x2x_attn); x2y_attn = self.softmax(x2y_attn); x2z_attn = self.softmax(x2z_attn)
            y2x_attn = self.softmax(y2x_attn); y2y_attn = self.softmax(y2y_attn); y2z_attn = self.softmax(y2z_attn)
            z2x_attn = self.softmax(z2x_attn); z2y_attn = self.softmax(z2y_attn); z2z_attn = self.softmax(z2z_attn)

        x2x_attn = self.attn_drop(x2x_attn); x2y_attn = self.attn_drop(x2y_attn); x2z_attn = self.attn_drop(x2z_attn)
        y2x_attn = self.attn_drop(y2x_attn); y2y_attn = self.attn_drop(y2y_attn); y2z_attn = self.attn_drop(y2z_attn)
        z2x_attn = self.attn_drop(z2x_attn); z2y_attn = self.attn_drop(z2y_attn); z2z_attn = self.attn_drop(z2z_attn)

        # .transpose(1, 2).reshape(B_, N, C//self.channel_scale)
        x2x_m = (x2x_attn @ x_v).reshape(B_, C//self.channel_scale, N).transpose(1, 2); x2y_m = (x2y_attn @ x_v).reshape(B_, C//self.channel_scale, N).transpose(1, 2); x2z_m = (x2z_attn @ x_v).reshape(B_, C//self.channel_scale, N).transpose(1, 2)
        y2x_m = (y2x_attn @ y_v).reshape(B_, C//self.channel_scale, N).transpose(1, 2); y2y_m = (y2y_attn @ y_v).reshape(B_, C//self.channel_scale, N).transpose(1, 2); y2z_m = (y2z_attn @ y_v).reshape(B_, C//self.channel_scale, N).transpose(1, 2)
        z2x_m = (z2x_attn @ z_v).reshape(B_, C//self.channel_scale, N).transpose(1, 2); z2y_m = (z2y_attn @ z_v).reshape(B_, C//self.channel_scale, N).transpose(1, 2); z2z_m = (z2z_attn @ z_v).reshape(B_, C//self.channel_scale, N).transpose(1, 2)
        x2x_m = self.proj_drop(self.proj(x2x_m)); x2y_m = self.proj_drop(self.proj(x2y_m)); x2z_m = self.proj_drop(self.proj(x2z_m))
        y2x_m = self.proj_drop(self.proj(y2x_m)); y2y_m = self.proj_drop(self.proj(y2y_m)); y2z_m = self.proj_drop(self.proj(y2z_m))
        z2x_m = self.proj_drop(self.proj(z2x_m)); z2y_m = self.proj_drop(self.proj(z2y_m)); z2z_m = self.proj_drop(self.proj(z2z_m))

        # x2x_e = x2x_m / (x2x_m + y2x_m + z2x_m); x2y_e = x2y_m / (x2y_m + y2y_m + z2y_m); x2z_e = x2z_m / (x2z_m + y2z_m + z2z_m)
        # y2x_e = y2x_m / (x2x_m + y2x_m + z2x_m); y2y_e = y2y_m / (x2y_m + y2y_m + z2y_m); y2z_e = y2z_m / (x2z_m + y2z_m + z2z_m)
        # z2x_e = z2x_m / (x2x_m + y2x_m + z2x_m); z2y_e = z2y_m / (x2y_m + y2y_m + z2y_m); z2z_e = z2z_m / (x2z_m + y2z_m + z2z_m)

        x = (x2x_m + y2x_m + z2x_m) / 3
        y = (x2y_m + y2y_m + z2y_m) / 3
        z = (x2z_m + y2z_m + z2z_m) / 3

        # x = x2x_m
        # y = y2y_m
        # z = z2z_m

        return x, y, z
    
    def forward(self, x, y, z):

        B, C, H, W = x.shape
        x_input = self.norm(x.permute(0, 2, 3, 1).view(B, H*W, C)).view(B, H, W, C)
        y_input = self.norm(y.permute(0, 2, 3, 1).view(B, H*W, C)).view(B, H, W, C)
        z_input = self.norm(z.permute(0, 2, 3, 1).view(B, H*W, C)).view(B, H, W, C)
        assert self.window_size[0] == self.window_size[1]

        x_windows = window_partition(x_input, self.window_size[0])  # nW*B, window_size, window_size, C
        y_windows = window_partition(y_input, self.window_size[0])
        z_windows = window_partition(z_input, self.window_size[0])
        x_windows = x_windows.view(-1, self.window_size[0] * self.window_size[1], C)  # nW*B, window_size*window_size, C
        y_windows = y_windows.view(-1, self.window_size[0] * self.window_size[1], C)
        z_windows = z_windows.view(-1, self.window_size[0] * self.window_size[1], C)

        x_windows, y_windows, z_windows = self.cal_WHA(x_windows, y_windows, z_windows, mask=None)

        x = window_reverse(x_windows, self.window_size[0], H, W).permute(0, 3, 1, 2)  # B, C, H, W
        y = window_reverse(y_windows, self.window_size[0], H, W).permute(0, 3, 1, 2)
        z = window_reverse(z_windows, self.window_size[0], H, W).permute(0, 3, 1, 2)

        return x , y , z

    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}'

def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias)

class MeanShift(nn.Conv2d):
    def __init__(self, rgb_range, rgb_mean, rgb_std, sign=-1):
        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1)
        self.weight.data.div_(std.view(3, 1, 1, 1))
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean)
        self.bias.data.div_(std)
        self.requires_grad = False

class BasicBlock(nn.Sequential):
    def __init__(
        self, in_channels, out_channels, kernel_size, stride=1, bias=False,
        bn=True, act=nn.ReLU(True)):

        m = [nn.Conv2d(
            in_channels, out_channels, kernel_size,
            padding=(kernel_size//2), stride=stride, bias=bias)
        ]
        if bn: m.append(nn.BatchNorm2d(out_channels))
        if act is not None: m.append(act)
        super(BasicBlock, self).__init__(*m)

class ResBlock(nn.Module):
    def __init__(
        self, conv, n_feat, kernel_size,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(ResBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(conv(n_feat, n_feat, kernel_size, bias=bias))
            if bn: m.append(nn.BatchNorm2d(n_feat))
            if i == 0: m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x

        return res

class DenseBlock(nn.Module):
    def __init__(self, channel_in, channel_out, init='xavier', gc=32, bias=True):
        super(DenseBlock, self).__init__()
        self.conv1 = nn.Conv2d(channel_in, gc, 3, 1, 1, bias=bias)
        self.conv2 = nn.Conv2d(channel_in + gc, gc, 3, 1, 1, bias=bias)
        self.conv3 = nn.Conv2d(channel_in + 2 * gc, gc, 3, 1, 1, bias=bias)
        self.conv4 = nn.Conv2d(channel_in + 3 * gc, gc, 3, 1, 1, bias=bias)
        self.conv5 = nn.Conv2d(channel_in + 4 * gc, channel_out, 3, 1, 1, bias=bias)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        # if init == 'xavier':
        #     mutil.initialize_weights_xavier([self.conv1, self.conv2, self.conv3, self.conv4], 0.1)
        # else:
        #     mutil.initialize_weights([self.conv1, self.conv2, self.conv3, self.conv4], 0.1)
        # mutil.initialize_weights(self.conv5, 0)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5

class Upsampler(nn.Sequential):
    def __init__(self, conv, scale, n_feat, bn=False, act=False, bias=True):

        m = []
        if (scale & (scale - 1)) == 0:    # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                m.append(conv(n_feat, 4 * n_feat, 3, bias))
                m.append(nn.PixelShuffle(2))
                if bn: m.append(nn.BatchNorm2d(n_feat))
                if act: m.append(act())
        elif scale == 3:
            m.append(conv(n_feat, 9 * n_feat, 3, bias))
            m.append(nn.PixelShuffle(3))
            if bn: m.append(nn.BatchNorm2d(n_feat))
            if act: m.append(act())
        else:
            raise NotImplementedError

        super(Upsampler, self).__init__(*m)


class Conv_up(nn.Module):
    def __init__(self, c_in, mid_c, up_factor):
        super(Conv_up, self).__init__()

        body = [nn.Conv2d(in_channels=c_in, out_channels=mid_c, kernel_size=3, padding=3 // 2), nn.ReLU(),
                # nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=3 // 2), nn.ReLU(),
                # nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=3 // 2), nn.ReLU(),
                ]
        self.body = nn.Sequential(*body)
        conv = default_conv
        ## x3 00
        ## x2 11
        if up_factor == 2:
            modules_tail = [
                # nn.ConvTranspose2d(64, 64, kernel_size=3, stride=up_factor, padding=1, output_padding=1),
                nn.Upsample(scale_factor=2),
                conv(mid_c, c_in,3),
                conv(c_in, c_in, 3)]

        elif up_factor == 3:
            modules_tail = [
                # nn.ConvTranspose2d(64, 64, kernel_size=3, stride=up_factor, padding=0, output_padding=0),
                nn.Upsample(scale_factor=3),
                conv(mid_c, c_in,3),
                conv(c_in, c_in, 3)]

        elif up_factor == 4:
            modules_tail = [
                # nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
                # nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.Upsample(scale_factor=4),
                conv(mid_c, c_in,3),
                conv(c_in, c_in, 3)]
        self.tail = nn.Sequential(*modules_tail)

    def forward(self, input):

        out = self.body(input)
        out = self.tail(out)
        return out


class Conv_down(nn.Module):
    def __init__(self, c_in,mid_c, up_factor):
        super(Conv_down, self).__init__()

        body = [nn.Conv2d(in_channels=c_in, out_channels=mid_c, kernel_size=3, padding=3 // 2), nn.ReLU(),
                # nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=3 // 2), nn.ReLU(),
                # nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=3 // 2), nn.ReLU(),
                ]
        self.body = nn.Sequential(*body)
        conv = default_conv
        if up_factor == 4:
            modules_tail = [
                # nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, stride=2),
                # nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, stride=2),
                nn.MaxPool2d(4),
                conv(mid_c, c_in,3),
                conv(c_in, c_in, 3)]

        elif up_factor == 3:
            modules_tail = [
                # nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, stride=up_factor),
                nn.MaxPool2d(3),
                conv(mid_c, c_in,3),
                conv(c_in, c_in, 3)]

        elif up_factor == 2:
            modules_tail = [
                # nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, stride=up_factor),
                nn.MaxPool2d(2),
                conv(mid_c, c_in,3),
                conv(c_in, c_in, 3)]
                
        self.tail = nn.Sequential(*modules_tail)

    def forward(self, input):

        out = self.body(input)
        out = self.tail(out)
        return out

class HinResBlock(nn.Module):
    def __init__(self, channel_in, channel_out):
        super(HinResBlock, self).__init__()
        feature = 64
        self.conv1 = nn.Conv2d(channel_in, feature, kernel_size=3, padding=1)
        self.relu1 = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.conv2 = nn.Conv2d(feature, feature, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d((feature+channel_in), channel_out, kernel_size=3, padding=1)
        self.norm = nn.InstanceNorm2d(feature // 2, affine=True)

    def forward(self, x):
        residual = self.relu1(self.conv1(x))

        out_1, out_2 = torch.chunk(residual, 2, dim=1)
        residual = torch.cat([self.norm(out_1), out_2], dim=1)

        residual = self.relu1(self.conv2(residual))
        input = torch.cat((x, residual), dim=1)
        out = self.conv3(input)
        return out


def subnet(net_structure, init='xavier'):
    def constructor(channel_in, channel_out):
        if net_structure == 'DBNet':
            if init == 'xavier':
                return DenseBlock(channel_in, channel_out, init)
            else:
                return DenseBlock(channel_in, channel_out)
        elif net_structure == 'Resnet':
            return ResBlock(channel_in, channel_out)
        elif net_structure == 'HinResnet':
            return HinResBlock(channel_in, channel_out)
        else:
            return None
    return constructor

class InvBlock(nn.Module):
    def __init__(self, subnet_constructor, channel_num1, channel_num2, clamp=0.8):
        super(InvBlock, self).__init__()
        # channel_num: 3
        # channel_split_num: 1

        self.split_len1 = channel_num1  # 1
        self.split_len2 = channel_num2  # 2

        self.clamp = clamp

        self.F = subnet_constructor(self.split_len2, self.split_len1)
        self.G = subnet_constructor(self.split_len1, self.split_len2)
        self.H = subnet_constructor(self.split_len1, self.split_len2)

        #in_channels = 3
        # self.invconv = InvertibleConv1x1(channel_num, LU_decomposed=True)
        # self.flow_permutation = lambda z, logdet, rev: self.invconv(z, logdet, rev)

    # def forward(self, x, rev=False):
    def forward(self, x, rev=False):
        x1, x2 = (x.narrow(1, 0, self.split_len1), x.narrow(1, self.split_len1, self.split_len2))
        if not rev:
            # invert1x1conv
            # x, logdet = self.flow_permutation(x, logdet=0, rev=False)

            # split to 1 channel and 2 channel.
            # x1, x2 = (x.narrow(1, 0, self.split_len1), x.narrow(1, self.split_len1, self.split_len2))

            y1 = x1 + self.F(x2)  # 1 channel
            self.s = self.clamp * (torch.sigmoid(self.H(y1)) * 2 - 1)
            y2 = x2.mul(torch.exp(self.s)) + self.G(y1)  # 2 channel
            
        else:
            # split.
            # x1, x2 = (x.narrow(1, 0, self.split_len1), x.narrow(1, self.split_len1, self.split_len2))
            self.s = self.clamp * (torch.sigmoid(self.H(x1)) * 2 - 1)
            y2 = (x2 - self.G(x1)).div(torch.exp(self.s))
            y1 = x1 - self.F(y2)

            # x = torch.cat((y1, y2), 1)
            # print("rev_inn")
            # inv permutation
            # out = x
        out = torch.cat((y1, y2), 1)
        return out