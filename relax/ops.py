import math
import pdb
from itertools import product
import torch
import torch.fft
from torch import nn
from torch.nn import functional as F
if int(torch.__version__.split('.')[1]) < 8:
    from torch_butterfly.complex_utils import complex_matmul
else:
    from torch import matmul as complex_matmul


def Conv(dims):
    '''returns PyTorch convolution module of specified dimension'''

    return getattr(nn, 'Conv'+str(dims)+'d')


def ConvTranspose(dims):
    '''returns PyTorch transposed convolution module of a specified dimension'''

    return getattr(nn, 'ConvTranspose'+str(dims)+'d')


def AvgPool(dims):
    '''returns PyTorch average pooling module of specified dimension'''

    return getattr(nn, 'AvgPool'+str(dims)+'d')


class Fourier(nn.Module):
    '''PyTorch module that applies some FFT to an input'''

    def __init__(self, inv=False, normalized=False, dims=1, compact=False, odd=False):
        '''
        Args:
            inv: if True use inverse FFT
            normalized: if True use normalized FFT
            dims: number of dimensions to transform
            compact: if True use compact "real" FFT
            odd: original signal has odd length in the last dimension (only uised if both inv and compact are True)
        '''

        super().__init__()
        self.inv = inv
        self.norm = 'ortho' if normalized else 'backward'
        self.dim = list(range(-dims, 0))
        self.compact = compact
        self.odd = odd and self.inv and self.compact

    def forward(self, input):

        if self.compact:
            func = torch.fft.irfftn if self.inv else torch.fft.rfftn
        else:
            func = torch.fft.ifftn if self.inv else torch.fft.fftn
        return func(input, norm=self.norm, dim=self.dim, s=input.shape[-len(self.dim):-1]+(2*input.shape[-1]-1,) if self.odd else None)


def int2tuple(int_or_tuple, length=2, allow_other=False):
    '''converts bools, ints, or slices to tuples of the specified length via repetition
    Args:
        int_or_tuple: bool, int, or slice, or a tuple of the same
        length: expected tuple length
        allow_other: allow tuple to have length other than 'length'
    '''

    if type(int_or_tuple) in {bool, int, slice, type(None)}:
        return tuple([int_or_tuple] * length)
    if len(int_or_tuple) != length and not allow_other:
        raise(ValueError("tuple must have length " + str(length)))
    return int_or_tuple


def multichannel_prod(x, w, separable=False, einsum=False):
    '''multiple-channel product element-wise product
    Args:
        x: batched data
        w: kernel weights
        separable: if True applies one kernel to each channel
        einsum: if True use torch.einsum for product
    '''

    if separable:
        x *= w.transpose(0, 1)
        return x
    
    dims = len(x.shape)-2
    if einsum:
        if dims > 3:
            raise(NotImplementedError("must have einsum=False if using >3 dims"))
        return torch.einsum('xyz'[:dims].join(['bi', ',io', '->bo', '']), x, w.transpose(0, 1))
    return complex_matmul(x.permute(*range(2, 2+dims), 0, 1), w.permute(*range(2, 2+dims), 1, 0)).permute(-2, -1, *range(dims))


class FNO(nn.Module):
    '''multi-dimensional reimplementation of Fourier Neural Operator [Li et al., ICLR 2021]'''

    @staticmethod
    def ifftn(*args, **kwargs):
        return torch.fft.ifftn(*args, **kwargs).real

    def __init__(self, in_channels, out_channels, modes, groups=1, compact=True, pad=False, einsum=False):
        '''
        Args:
            in_channels: number of input channels
            out_channels: number of output channels
            modes: number of modes, i.e. size of spectral kernel
            groups: number of channel groups
            compact: if True use compact "real" FFT
            pad: if True pad input to a power of 2 before applying FFT
            einsum: if True use torch.einsum for product
        '''

        super(FNO, self).__init__()
        if groups > 1:
            if in_channels % groups:
                raise(ValueError("in_channels must be divisible by groups"))
            elif in_channels != out_channels or groups != in_channels:
                raise(NotImplementedError("in_channels must equal groups"))
        self.groups = groups
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dims = 1 if type(modes) == int else len(modes)
        self.modes = int2tuple(modes, length=self.dims)
        self.scale = 1. / (in_channels * out_channels)
        self.compact = compact
        self.pad = pad
        self.fft = torch.fft.rfftn if self.compact else torch.fft.fftn
        self.ifft = torch.fft.irfftn if self.compact else self.ifftn
        kernel_size = [2*m for m in self.modes[:-1]] + [(2-self.compact)*self.modes[-1]]
        self.weight = nn.Parameter(self.scale * torch.rand(out_channels, in_channels//groups, *kernel_size, dtype=torch.complex64))
        self.einsum = einsum

    def _get_slices(self, size=None):

        if self.compact:
            size = [2*m for m in self.modes[:-1]] if size is None else size[:-1]
            modes = self.modes[:-1]
            end = [slice(self.modes[-1])]
        else:
            size = [2*m for m in self.modes] if size is None else size
            modes = self.modes
            end = []
        for bits in product(*[range(2)] * (self.dims-self.compact)):
            yield [slice(None), slice(None)] + [slice(n-m, n) if b else slice(m) for b, n, m in zip(reversed(bits), size, modes)] + end

    def forward(self, x):

        if self.pad:
            unpad = [slice(None)] * 2 + [slice(s) for s in x.shape[2:]]
            s = [2 ** math.ceil(math.log2(s)) for s in x.shape[2:]]
            size = s[:-1] + [s[-1]//2+1 if self.compact else s[-1]]
        else:
            unpad, s = [slice(None)], x.shape[2:]
            size = list(x.shape[2:-1]) + [x.shape[-1]//2+1 if self.compact else x.shape[-1]]

        x_ft = self.fft(x, s=s, dim=tuple(range(-self.dims, 0)), norm='ortho')
        out_ft = torch.zeros(len(x), self.out_channels, *size, dtype=torch.complex64, device=x.device)
        for xslices, wslices in zip(self._get_slices(size), self._get_slices()):
            out_ft[xslices] = multichannel_prod(x_ft[xslices], self.weight[wslices], separable=self.groups == self.weight.shape[0] > 1, einsum=self.einsum)
        return self.ifft(out_ft, s=s if self.compact else None, dim=tuple(range(-self.dims, 0)), norm='ortho')[unpad]
