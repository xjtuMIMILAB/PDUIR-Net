import torch
import numpy as np
import os
sqrt = np.sqrt
import torch.nn.functional as F
from torch.fft import ifft, fft
import torchvision.transforms as T


def ifftshift(x, axes=None):
    assert torch.is_tensor(x) == True
    if axes is None:
        axes = tuple(range(x.ndim))
        shift = [-(dim // 2) for dim in x.shape]
    elif isinstance(axes, int):
        shift = -(x.shape[axes] // 2)
    else:
        shift = [-(x.shape[axis] // 2) for axis in axes]
    return torch.roll(x, shift, axes)


def fftshift(x, axes=None):
    assert torch.is_tensor(x) == True
    if axes is None:
        axes = tuple(range(x.ndim()))
        shift = [dim // 2 for dim in x.shape]
    elif isinstance(axes, int):
        shift = x.shape[axes] // 2
    else:
        shift = [x.shape[axis] // 2 for axis in axes]
    return torch.roll(x, shift, axes)

def ifft2c(x):
    device = x.device
    # nb, nc, nx, ny = x.size()
    ny = torch.Tensor([x.shape[-1]])
    ny = ny.to(device)
    nx = torch.Tensor([x.shape[-2]])
    nx = nx.to(device)
    x = ifftshift(x, axes=-2)
    x = torch.transpose(x, -2, -1)
    x = ifft(x)
    x = torch.transpose(x, -2, -1)
    x = torch.mul(fftshift(x, axes=-2), torch.sqrt(nx))
    x = ifftshift(x, axes=-1)
    x = ifft(x)
    x = torch.mul(fftshift(x, axes=-1), torch.sqrt(ny))
    return x


def fft2c(x):
    device = x.device
    # nb, nc, nx, ny = x.size()
    ny = torch.Tensor([x.shape[-1]]).to(device)
    nx = torch.Tensor([x.shape[-2]]).to(device)
    x = ifftshift(x, axes=-2)
    x = torch.transpose(x, -2, -1)
    x = fft(x)
    x = torch.transpose(x, -2, -1)
    x = torch.div(fftshift(x, axes=-2), torch.sqrt(nx))
    x = ifftshift(x, axes=-1)
    x = fft(x)
    x = torch.div(fftshift(x, axes=-1), torch.sqrt(ny))
    return x

def fft1c(x, dim):
    device = x.device
    # nb, nt, nx, ny = x.size()
    nt = torch.Tensor([x.shape[dim]]).to(device)
    x = ifftshift(x, axes=dim)
    x = torch.transpose(x, dim, -1)
    x = fft(x)
    x = torch.transpose(x, dim, -1)
    x = torch.div(fftshift(x, axes=dim), torch.sqrt(nt))
    return x

def ifft1c(x, dim):
    device = x.device
    # nb, nt, nx, ny = x.size()
    nt = torch.Tensor([x.shape[dim]]).to(device)
    x = ifftshift(x, axes=dim)
    x = torch.transpose(x, dim, -1)
    x = ifft(x)
    x = torch.transpose(x, dim, -1)
    x = torch.mul(fftshift(x, axes=dim), torch.sqrt(nt))
    return x

def r2c(x):
    re, im = torch.chunk(x,2,1)
    x = torch.complex(re,im)
    return x.squeeze(1)

def c2r(x):
    # 这里有疑问 关于维度的变换！！！
    x = x.unsqueeze(1)
    x = torch.cat([torch.real(x),torch.imag(x)],1)
    return x


def ssos(x):
    xr, xi = torch.chunk(x,2,1)
    x = torch.pow(torch.abs(xr),2)+torch.pow(torch.abs(xi),2)
    x = torch.sum(x, dim=1)
    x = torch.pow(x,0.5)
    # x = torch.unsqueeze(x,1)
    return x

def sos(x):
    # x = r2c(x)
    x = torch.pow(torch.abs(x),2)
    x = torch.sum(x, dim=1)
    x = torch.pow(x,0.5)
    # x = torch.unsqueeze(x,1)
    return x

def dot_batch(x1, x2):
    batch = x1.shape[0]
    res = torch.reshape(x1 * x2, (batch, -1))
    # res = torch.reshape(x1 * x2, (-1, 1))
    return torch.sum(res, 1)

def readcfl(name):
    # get dims from .hdr
    h = open(name + ".hdr", "r")
    h.readline() # skip
    l = h.readline()
    h.close()
    dims = [int(i) for i in l.split( )]

    # remove singleton dimensions from the end
    n = np.prod(dims)
    dims_prod = np.cumprod(dims)
    dims = dims[:np.searchsorted(dims_prod, n)+1]

    # load data and reshape into dims
    d = open(name + ".cfl", "r")
    a = np.fromfile(d, dtype=np.complex64, count=n);
    d.close()
    return a.reshape(dims, order='F') # column-major


def writecfl(name, array):
    h = open(name + ".hdr", "w")
    h.write('# Dimensions\n')
    for i in (array.shape):
            h.write("%d " % i)
    h.write('\n')
    h.close()
    d = open(name + ".cfl", "w")
    array.T.astype(np.complex64).tofile(d) # tranpose for column-major order
    d.close()
    
    
    
def norm(x:torch.Tensor):
    #  group norm
    b,c,h,w = x.shape
    x= x.view(b,2,c//2 *h*w)
    mean = x.mean(dim = 2).view(b,2,1,1)
    std = x.std(dim=2).view(b,2,1,1)
    x =  x.view(b,c,h,w)
    
    return (x - mean) /std,mean,std
def unnorm(x,mean,std):
    return x *std + mean 



