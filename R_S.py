
# from einops import rearrange
import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
from utils import fft2c, ifft2c, r2c, c2r,writecfl,norm,unnorm
import math
import numpy as np

import torch
import  torch.nn as nn
import fastmri
from abc import abstractmethod

import math
import mdi_function as mdi
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F

from nn import (
    checkpoint,
    conv_nd,
    linear,
    avg_pool_nd,
    zero_module,
    normalization,
    timestep_embedding,
)
def calculate_S(S2,Nc,Na,N2):
    a = torch.zeros_like(S2[...,0,0,0], dtype=torch.complex64)
    b = torch.zeros_like(S2[...,0,0,0], dtype=torch.float32)
    
    for iter_coil in range(Nc):
        for iter_fa in range(Na):
            for iter_echo in range(N2 - 1):
                a += torch.conj(S2[..., iter_coil, iter_echo, iter_fa]) * S2[..., iter_coil, iter_echo + 1, iter_fa]
                b += torch.abs(S2[..., iter_coil, iter_echo, iter_fa]) ** 2

    delta_S = a / (b + 1e-8)  # signal ratio between neighboring echoes
    return delta_S

def calculate_R(S1,S2,Nc,N2):
    A, B = 0, 0
    for coil in range(Nc):
        A += torch.conj(S1[..., coil, 0, 1]) * S1[..., coil, 0, 0]
        B += torch.abs(S1[..., coil, 0, 1]) ** 2

    R1 = A / (B + 1e-8)
    
    A, B = 0, 0
    for coil in range(Nc):
        for TE in range(N2):
            A += torch.conj(S2[..., coil, TE, 1]) * S2[..., coil, TE, 0]
            B += torch.abs(S2[..., coil, TE, 1]) ** 2
    R2 = A / (B + 1e-8) 

    return R1,R2

def norm_r1(R1_map,max_magnitude=3.0):
    magnitude = torch.abs(R1_map)
    scale = torch.where(magnitude > max_magnitude, 
                       max_magnitude / magnitude, 
                       torch.ones_like(magnitude))
    return R1_map * scale
from unet import Unet
class update_RS(nn.Module):
    def __init__(self, ):
        super().__init__()
        self.r1 = Unet(in_chans=2 ,out_chans=2)
        self.r2 = Unet(in_chans=2 ,out_chans=2)
        self.s = Unet(in_chans=2 ,out_chans=2)
    def forward(self,R1,R2,S):
        S_mapping = self.s(S) + S
        R1_mapping = self.r1(R1) + R1
        R2_mapping = self.r2(R2) + R2
        return R1_mapping,R2_mapping,S_mapping
        
class update_RS_Z(nn.Module):
    def __init__(self, ):
        super().__init__()
        
        self.beta1 = nn.Parameter(torch.tensor([0.1]), requires_grad=True)
        self.beta2 = nn.Parameter(torch.tensor([0.1]), requires_grad=True)
        self.beta3 = nn.Parameter(torch.tensor([0.1]), requires_grad=True)
        self.mu1 = nn.Parameter(torch.tensor([0.1]), requires_grad=True)
        self.mu2 = nn.Parameter(torch.tensor([0.1]), requires_grad=True)
        self.mu3 = nn.Parameter(torch.tensor([0.1]), requires_grad=True)
        self.mu4 = nn.Parameter(torch.tensor([0.1]), requires_grad=True)
        self.r1 = Unet(in_chans=2 ,out_chans=2)
        self.r2 = Unet(in_chans=2 ,out_chans=2)
        self.s = Unet(in_chans=2 ,out_chans=2)
    def sens_expand(self, x: torch.Tensor, sens_maps: torch.Tensor) -> torch.Tensor: 
    # function F* E^-1
        _, c, _, _, _ = sens_maps.shape 
        return fastmri.fft2c(fastmri.complex_mul(x.unsqueeze(1).repeat(1, c, 1, 1, 1), sens_maps))
    

    # 
    def sens_reduce(self, x: torch.Tensor, sens_maps: torch.Tensor) -> torch.Tensor:
    # function  E * F^-1 
        x = fastmri.ifft2c(x)

        return fastmri.complex_mul(x, fastmri.complex_conj(sens_maps)).sum( dim=1, keepdim=False)

    def complex2real(self,x:torch.Tensor):
        x_real = x[...,0].squeeze(-1)
        x_imag = x[...,1].squeeze(-1)
        return x_real + 1j * x_imag
    
    
    def real2complex(self,x:torch.Tensor):
        x_real = x.real
        x_imag = x.imag
        return torch.stack((x_real,x_imag),dim = -1)
    
    def update_RS(self,R1,R2,S):
        
        S_mapping = self.s(S) + S
        R1_mapping = self.r1(R1) + R1
        R2_mapping = self.r2(R2) + R2
        
        
        return R1_mapping,R2_mapping,S_mapping
    
    def update_Z(self,img,S_map,R1_map,R2_map,Z,mask_img):
        if not Z.requires_grad:
            Z.requires_grad = True
        img = img.permute(0,1,4,2,3)
        # var1, var2, var3, var4, var5, var6,var7, var8, var9, var10, var11, var12 = [chunk.squeeze(1) for chunk in torch.chunk(Z, chunks=12, dim=1)]  
        R1,R2,S = mdi.R_S(Z)
        
        R1 = R1 * mask_img
        R2 = R2 * mask_img
        S = S * mask_img 
        R1 = norm_r1(R1)
        R1 = torch.stack((R1.real,R1.imag),dim=1)
        R2 = torch.stack((R2.real,R2.imag),dim=1)
        S = torch.stack((S.real,S.imag),dim=1)
        
        
        L_R1 = torch.sum((R1-R1_map)**2)
        L_R2 = torch.sum((R2-R2_map)**2)
        L_S = torch.sum((S-S_map)**2)
        
        grad_1 = torch.autograd.grad(outputs=L_R1, inputs=Z,  # type: ignore
                                    grad_outputs=torch.ones_like(L_R1), retain_graph=True)[0] # type: ignore
        grad_2 = torch.autograd.grad(outputs=L_R2, inputs=Z,  # type: ignore
                                    grad_outputs=torch.ones_like(L_R2), retain_graph=True)[0] # type: ignore
        grad_3 = torch.autograd.grad(outputs=L_S, inputs=Z, 
                                    grad_outputs=torch.ones_like(L_S), retain_graph=True)[0]
        
        grad_1 = torch.nan_to_num(grad_1, nan=0.0)
        grad_2 = torch.nan_to_num(grad_2, nan=0.0)
        grad_3 = torch.nan_to_num(grad_3, nan=0.0)
        grads = self.mu1 * grad_1+ self.mu2 * grad_2 + self.mu3 * grad_3 + self.mu4 * (Z-img)
        Z = Z - grads
        
        return Z
    def forward(self,mask_img,img,Z):
        
        Z = img.permute(0,1,4,2,3)
        
        
        R1_map_2,R2_map_2,S_map_2 = mdi.R_S(Z)
            
        R1_map_2 = R1_map_2 * mask_img
        R2_map_2 = R2_map_2 * mask_img
        S_map_2 = S_map_2 * mask_img   
        R1_map_2 =  norm_r1(R1_map_2) 
        R1_map = torch.stack((R1_map_2.real,R1_map_2.imag),dim=1)
        R2_map = torch.stack((R2_map_2.real,R2_map_2.imag),dim=1)
        S_map = torch.stack((S_map_2.real,S_map_2.imag),dim=1)
        
        # savemat('RS_recon.mat',{'S':S_map.detach().cpu().numpy(),
        #                            'R1':R1_map.detach().cpu().numpy(),
        #                            'R2':R2_map.detach().cpu().numpy(),
        #                            })
        
        R1_map,R2_map,S_map = self.update_RS(R1_map,R2_map,S_map)
        
        # with torch.enable_grad():
        #     Z = self.update_Z(img,S_map,R1_map,R2_map,Z,mask_img)
        return R1_map,R2_map,S_map

class update_RS_Z_2(nn.Module):
    def __init__(self, ):
        super().__init__()
        
        self.beta1 = nn.Parameter(torch.tensor([0.1]), requires_grad=True)
        self.beta2 = nn.Parameter(torch.tensor([0.1]), requires_grad=True)
        self.beta3 = nn.Parameter(torch.tensor([0.1]), requires_grad=True)
        self.mu1 = nn.Parameter(torch.tensor([0.1]), requires_grad=True)
        self.mu2 = nn.Parameter(torch.tensor([0.1]), requires_grad=True)
        self.mu3 = nn.Parameter(torch.tensor([0.1]), requires_grad=True)
        self.mu4 = nn.Parameter(torch.tensor([0.1]), requires_grad=True)
        self.net_rs = UNetModel(
            image_size =  [288,84],
                in_channels = 12,
                model_channels = 32,
                out_channels = 12,
                num_res_blocks = 1,
                attention_resolutions = [1],
                dropout=0,
                channel_mult=(1,2,4),
                conv_resample=True,
                dims=2,
                num_classes=None,
                use_checkpoint=False,
                use_fp16=False,
                num_heads=1,
                num_head_channels=-1,
                num_heads_upsample=-1,
                use_scale_shift_norm=False,
                resblock_updown=False,
                use_new_attention_order=False,
            )
        
    def sens_expand(self, x: torch.Tensor, sens_maps: torch.Tensor) -> torch.Tensor: 
    # function F* E^-1
        _, c, _, _, _ = sens_maps.shape 
        return fastmri.fft2c(fastmri.complex_mul(x.unsqueeze(1).repeat(1, c, 1, 1, 1), sens_maps))
    

    # 
    def sens_reduce(self, x: torch.Tensor, sens_maps: torch.Tensor) -> torch.Tensor:
    # function  E * F^-1 
        x = fastmri.ifft2c(x)

        return fastmri.complex_mul(x, fastmri.complex_conj(sens_maps)).sum( dim=1, keepdim=False)

    def complex2real(self,x:torch.Tensor):
        x_real = x[...,0].squeeze(-1)
        x_imag = x[...,1].squeeze(-1)
        return x_real + 1j * x_imag
    
    
    def real2complex(self,x:torch.Tensor):
        x_real = x.real
        x_imag = x.imag
        return torch.stack((x_real,x_imag),dim = -1)
    
    def update_RS(self,img):
        

        R1,R2,S = mdi.R_S(img)
        R1 = torch.stack((R1.real,R1.imag),dim=1)
        R2 = torch.stack((R2.real,R2.imag),dim=1)
        S = torch.stack((S.real,S.imag),dim=1)
        S_mapping , R_mapping = self.net_rs(S,R1,R2)
        R1_map,R2_map = torch.chunk(R_mapping,chunks=2,dim=1)
              
        return R1_map,R2_map,S_mapping

    def forward(self,img):
        
        R1_map,R2_map,S_map = self.update_RS(img)
        
        return R1_map,R2_map,S_map
    
    
    
from unet import Unet
class update_RS_Z_3(nn.Module):
    def __init__(self, ):
        super().__init__()
        self.r1 = Unet(in_chans = 2,out_chans=2)
        self.r2 = Unet(in_chans = 2,out_chans=2)
        self.s = convnet()


    
    def update_RS(self,img,mask):
        

        R1,R2,S = mdi.R_S(img)
        R1 = R1 * mask
        R2 = R2 * mask
        S =S * mask
        # from scipy.io import savemat
        # savemat('RS_imput.mat',{'R1':R1.detach().cpu().numpy(),'R2':R2.detach().cpu().numpy()})
        R1 = torch.stack((R1.real,R1.imag),dim=1)
        R2 = torch.stack((R2.real,R2.imag),dim=1)
        S = torch.stack((S.real,S.imag),dim=1)
        
        # R1,R1_norm,R1_std = norm(R1)
        # R2,R2_norm,R1_std = norm(R2)
        # S,S_norm,R1_std = norm(S)
        
        S_mapping = self.s(S) + S
        R1_mapping = self.r1(R1) + R1
        R2_mapping = self.r2(R2) + R2
        # R1_mapping = unnorm(R1_mapping,R1_norm,R1_std)
        # R2_mapping = unnorm(R2_mapping,R2_norm,R1_std)
        # S_mapping = unnorm(S_mapping,S_norm,R1_std)
        # S = unnorm(S,S_norm,R1_std)
        # return S_mapping,S
        # return R1_mapping
        return R1_mapping,R2_mapping,S_mapping

    def forward(self,img,mask):
        R1_mapping = self.update_RS(img,mask)
        # S_map,S = self.update_RS(img,mask)
        # R1_map,R2_map,S_map = self.update_RS(img,mask)
        
        # return S_map,S
        return R1_mapping
        # return R1_map,R2_map,S_map
class convnet(nn.Module):
    def __init__(self) -> None:
        super(convnet,self).__init__()
        self.conv1 = nn.Conv2d(2,8,kernel_size=3,padding=1)
        self.conv2 = nn.Conv2d(8,32,kernel_size=3,padding=1)
        self.conv3 = nn.Conv2d(32,64,kernel_size=3,padding=1)
        self.conv4 = nn.Conv2d(64,32,kernel_size=3,padding=1)
        self.conv5 = nn.Conv2d(32,8,kernel_size=3,padding=1)
        self.conv6 = nn.Conv2d(8,2,kernel_size=3,padding=1)
        self.relu = nn.ReLU()
        
    def forward(self,x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        x = self.relu(self.conv5(x))
        x = self.conv6(x)
        return x
    
class Upsample(nn.Module):
    """
    An upsampling layer with an optional convolution.

    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 upsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, dims=2, out_channels=None):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        if use_conv:
            self.conv = conv_nd(dims, self.channels, self.out_channels, 3, padding=1)

    def forward(self, x):
        assert x.shape[1] == self.channels
        if self.dims == 3:
            x = F.interpolate(
                x, (x.shape[2], x.shape[3] * 2, x.shape[4] * 2), mode="nearest"
            )
        else:
            x = F.interpolate(x, scale_factor=2, mode="nearest")
        if self.use_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    """
    A downsampling layer with an optional convolution.

    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 downsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, dims=2, out_channels=None):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        stride = 2 if dims != 3 else (1, 2, 2)
        if use_conv:
            self.op = conv_nd(
                dims, self.channels, self.out_channels, 3, stride=stride, padding=1
            )
        else:
            assert self.channels == self.out_channels
            self.op = avg_pool_nd(dims, kernel_size=stride, stride=stride)

    def forward(self, x):
        assert x.shape[1] == self.channels
        return self.op(x)

class ResBlock(nn.Module):
    def __init__(
        self,
        channels,
        dropout,
        out_channels=None,
        use_conv=False,
        use_scale_shift_norm=False,
        dims=2,
        use_checkpoint=False,
        up=False,
        down=False,
    ):
        super().__init__()
        self.channels = channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_checkpoint = use_checkpoint
        self.use_scale_shift_norm = use_scale_shift_norm

        self.in_layers = nn.Sequential(
            normalization(channels),
            nn.SiLU(),
            conv_nd(dims, channels, self.out_channels, 3, padding=1),
        )

        self.updown = up or down

        if up:
            self.h_upd = Upsample(channels, False, dims)
            self.x_upd = Upsample(channels, False, dims)
        elif down:
            self.h_upd = Downsample(channels, False, dims)
            self.x_upd = Downsample(channels, False, dims)
        else:
            self.h_upd = self.x_upd = nn.Identity()

        self.out_layers = nn.Sequential(
            normalization(self.out_channels),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            zero_module(
                conv_nd(dims, self.out_channels, self.out_channels, 3, padding=1)
            ),
        )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = conv_nd(
                dims, channels, self.out_channels, 3, padding=1
            )
        else:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, 1)

    def forward(self, x):
        """
        Apply the block to a Tensor.

        :param x: an [N x C x ...] Tensor of features.
        :return: an [N x C_out x ...] Tensor of outputs.
        """
        if self.use_checkpoint:
            pass
        else:
            return self._forward(x)

    def _forward(self, x):
        if self.updown:
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
            h = in_rest(x)
            h = self.h_upd(h)
            x = self.x_upd(x)
            h = in_conv(h)
        else:
            h = self.in_layers(x)
        
        h = self.out_layers(h)
        return self.skip_connection(x) + h

def count_flops_attn(model, _x, y):
    """
    A counter for the `thop` package to count the operations in an
    attention operation.
    Meant to be used like:
        macs, params = thop.profile(
            model,
            inputs=(inputs, timestamps),
            custom_ops={QKVAttention: QKVAttention.count_flops},
        )
    """
    b, c, *spatial = y[0].shape
    num_spatial = int(np.prod(spatial))
    # We perform two matmuls with the same number of ops.
    # The first computes the weight matrix, the second computes
    # the combination of the value vectors.
    matmul_ops = 2 * b * (num_spatial ** 2) * c
    model.total_ops += th.DoubleTensor([matmul_ops])
class QKVAttentionLegacy(nn.Module):
    """
    A module which performs QKV attention. Matches legacy QKVAttention + input/ouput heads shaping
    """

    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv):
        """
        Apply QKV attention.

        :param qkv: an [N x (H * 3 * C) x T] tensor of Qs, Ks, and Vs.
        :return: an [N x (H * C) x T] tensor after attention.
        """
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.reshape(bs * self.n_heads, ch * 3, length).split(ch, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = th.einsum(
            "bct,bcs->bts", q * scale, k * scale
        )  # More stable with f16 than dividing afterwards
        weight = th.softmax(weight.float(), dim=-1).type(weight.dtype)
        a = th.einsum("bts,bcs->bct", weight, v)
        return a.reshape(bs, -1, length)

    @staticmethod
    def count_flops(model, _x, y):
        return count_flops_attn(model, _x, y)


class QKVAttention(nn.Module):
    """
    A module which performs QKV attention and splits in a different order.
    """

    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv):
        """
        Apply QKV attention.

        :param qkv: an [N x (3 * H * C) x T] tensor of Qs, Ks, and Vs.
        :return: an [N x (H * C) x T] tensor after attention.
        """
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.chunk(3, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = th.einsum(
            "bct,bcs->bts",
            (q * scale).view(bs * self.n_heads, ch, length),
            (k * scale).view(bs * self.n_heads, ch, length),
        )  # More stable with f16 than dividing afterwards
        weight = th.softmax(weight.float(), dim=-1).type(weight.dtype)
        a = th.einsum("bts,bcs->bct", weight, v.reshape(bs * self.n_heads, ch, length))
        return a.reshape(bs, -1, length)

    @staticmethod
    def count_flops(model, _x, y):
        return count_flops_attn(model, _x, y)
class AttentionBlock(nn.Module):
    def __init__(
        self,
        channels,
        num_heads=1,
        num_head_channels=-1,
        use_checkpoint=False,
        use_new_attention_order=False,
    ):
        super().__init__()
        self.channels = channels
        if num_head_channels == -1:
            self.num_heads = num_heads
        else:
            assert (
                channels % num_head_channels == 0
            ), f"q,k,v channels {channels} is not divisible by num_head_channels {num_head_channels}"
            self.num_heads = channels // num_head_channels
        self.use_checkpoint = use_checkpoint
        self.norm = normalization(channels)
        self.qkv = conv_nd(1, channels, channels * 3, 1)
        if use_new_attention_order:
            # split qkv before split heads
            self.attention = QKVAttention(self.num_heads)
        else:
            # split heads before split qkv
            self.attention = QKVAttentionLegacy(self.num_heads)

        self.proj_out = zero_module(conv_nd(1, channels, channels, 1))

    def forward(self, x):
        return checkpoint(self._forward, (x,), self.parameters(), True)

    def _forward(self, x):
        b, c, *spatial = x.shape
        x = x.reshape(b, c, -1)
        qkv = self.qkv(self.norm(x))
        h = self.attention(qkv)
        h = self.proj_out(h)
        return (x + h).reshape(b, c, *spatial)



class UNetModel(nn.Module):
    def __init__(
        self,
        image_size,
        in_channels,
        model_channels,
        out_channels,
        num_res_blocks,
        attention_resolutions,
        dropout=0,
        channel_mult=(1, 2, 4, 8),
        conv_resample=True,
        dims=2,
        num_classes=None,
        use_checkpoint=False,
        use_fp16=False,
        num_heads=1,
        num_head_channels=-1,
        num_heads_upsample=-1,
        use_scale_shift_norm=False,
        resblock_updown=False,
        use_new_attention_order=False,
        mode = 'bilinear',
        align_corners = False
    ):
        super().__init__()

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        self.image_size = image_size
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.num_classes = num_classes
        self.use_checkpoint = use_checkpoint
        self.dtype = th.float16 if use_fp16 else th.float32
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample
        self.mode = mode
        self.align_corners = align_corners
        
        
        ch = input_ch = int(channel_mult[0] * model_channels)
        self.input_S = nn.ModuleList(
            [nn.Sequential(conv_nd(dims, 2, ch, 3, padding=1))]
        )
        self.input_R = nn.ModuleList(
            [nn.Sequential(conv_nd(dims, 4, ch, 3, padding=1))]
        )
        self._feature_size = ch
        input_block_chans = [ch]
        ds = 0
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [
                    ResBlock(
                        ch,
                        dropout,
                        out_channels=int(mult * model_channels),
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = int(mult * model_channels)

                self.input_S.append(nn.Sequential(*layers))
                self.input_R.append(nn.Sequential(*layers))
                self._feature_size += ch
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_S.append(
                    nn.Sequential(
                        ResBlock(
                            ch,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            down=True,
                        )
                        if resblock_updown
                        else Downsample(
                            ch, conv_resample, dims=dims, out_channels=out_ch
                        )
                    )
                )
                self.input_R.append(
                    nn.Sequential(
                        ResBlock(
                            ch,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            down=True,
                        )
                        if resblock_updown
                        else Downsample(
                            ch, conv_resample, dims=dims, out_channels=out_ch
                        )
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                ds *= 2
                self._feature_size += ch

        self._feature_size += ch

        self.output_S = nn.ModuleList([])
        self.output_R = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                ich = input_block_chans.pop()
                layers = [
                    ResBlock(
                        ch + ich,
                        dropout,
                        out_channels=int(model_channels * mult),
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = int(model_channels * mult)
                
                if level and i == num_res_blocks:
                    out_ch = ch
                    layers.append(
                        ResBlock(
                            ch,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            up=True,
                        )
                        if resblock_updown
                        else Upsample(ch, conv_resample, dims=dims, out_channels=out_ch) # type: ignore
                    )
                    ds //= 2
                self.output_S.append(nn.Sequential(*layers))
                self.output_R.append(nn.Sequential(*layers))
                self._feature_size += ch

 
        self.S = nn.Sequential(
            normalization(ch),
            nn.SiLU(),
            zero_module(conv_nd(dims, input_ch, 2, 3, padding=1)),
        )
    
        self.R = nn.Sequential(
            normalization(ch),
            nn.SiLU(),
            zero_module(conv_nd(dims, input_ch, 4, 3, padding=1)),
        )
        
        
    def com2real(self,data):
        if not data.is_complex():
            data = data.to(torch.complex64)
        real =data.real
        imag = data.imag
        return torch.stack((real,imag),-1)
    def real2com(self,data):
        real =data[...,0].squeeze(-1)
        imag = data[...,1].squeeze(-1)
        return real+ 1j * imag
    
    def forward(self,S_map,R1_map,R2_map):
        # if not S_map.is_complex():
        #     S_map = S_map.to(torch.complex64)
        # if not R1_map.is_complex():
        #     R1_map = R1_map.to(torch.complex64)
        # if not R2_map.is_complex():
        #     R2_map = R2_map.to(torch.complex64)
        # S_map = self.com2real(S_map)# .permute(0,3,1,2)
        # R1_map = self.com2real(R1_map)# .permute(0,3,1,2)
        # R2_map = self.com2real(R2_map)# .permute(0,3,1,2)
        R_map = torch.concat((R1_map,R2_map),dim=1)
        # S_mapping
        hs_S = []
        h_S = S_map.type(self.dtype)
        for module in self.input_S:
            h_S = module(h_S)
            hs_S.append(h_S)        
        for module in self.output_S:
            h_S = th.cat([h_S, hs_S.pop()], dim=1)
            h_S = module(h_S)
        h_S = h_S.type(S_map.dtype)
        S_mapping = self.S(h_S)    # [nb, 2, ny nz]
        
        hs_R = []
        h_R = R_map.type(self.dtype)
        for module in self.input_R:
            h_R = module(h_R)
            hs_R.append(h_R)        
        for module in self.output_R:
            h_R = th.cat([h_R, hs_R.pop()], dim=1)
            h_R = module(h_R)
        h_R = h_R.type(R_map.dtype)
        R_mapping = self.R(h_R)    # [nb, 2, ny nz]
        
        
        return S_mapping , R_mapping