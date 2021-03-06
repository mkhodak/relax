import math
import pdb
from functools import lru_cache
from itertools import product
import torch
from torch import nn
from torch.nn import functional as F
from relax.ops import AvgPool, Conv, ConvTranspose, FNO, Fourier, SharedOperation, int2tuple, multichannel_prod


class TensorProduct(nn.Module):
    '''applies provided maps to multi-dimensional inputs'''
    def __init__(self, *maps):
        '''
        Args:
            maps: one argument for each dimension
        '''

        super().__init__()
        self.maps = nn.ModuleList(list(maps))

    def forward(self, x):

        for i, m in enumerate(reversed(self.maps)):
            x = m(x.transpose(-1, -i-1)).transpose(-1, -i-1)
        return x


class Pad(nn.Module):
    '''module that pads an input before convolution and generates a tuple for unpadding after'''

    def __init__(self, dims, in_size=None, kernel_size=1, dilation=1, padding=None, padding_mode='circular', power=False):
        '''
        Args:
            dims: number of padding dimensions
            in_size: expected input size before padding
            kernel_size: kernel size used by target convolution
            dilation: dilation rate used by target convolution
            padding: amount of zero padding
            padding_mode: type of padding used by target convolution
            power: enforce input size to be a power of 2
        '''

        super(Pad, self).__init__()
        self.in_size = ()
        self.pad = ()
        self.mode = 'constant' if padding_mode == 'zeros' else 'circular'
        self.unpad = []
        for d, k, n, p in zip(int2tuple(dilation, length=dims), 
                              int2tuple(kernel_size, length=dims),
                              int2tuple(in_size, length=dims),
                              int2tuple(padding, length=dims)):
            if padding_mode == 'zeros':
                p = 0 if p is None else p
                self.pad += (p, p)
                n = 2 ** math.ceil(math.log2(n+2*p)) if power else n
                p = 0
            else:
                self.pad += (0, 0)
                n = 2 ** math.ceil(math.log2(n)) if power else n
            self.in_size += (n,)
            if p is None:
                self.unpad.append((0, 0))
            else:
                a, b = (d*(k-1)) // 2 - p, 0 - (d*(k-1)+1) // 2 + p
                if a < 0 or (not n is None and (a > n+b)):
                    raise(ValueError("invalid padding"))
                self.unpad.append((a, b))

    def forward(self, x):

        x = F.pad(x, self.pad)
        pad, unpad = (), [slice(None), slice(None)]
        for xn, n, (a, b) in zip(x.shape[2:], self.in_size, self.unpad):
            pad = (0, 0 if n is None else n-xn) + pad
            unpad.append(slice(a, xn+b))
        return F.pad(x, pad), unpad


class Unpad(nn.Module):
    '''module that unpads input to required size given an unpadding tuple generated by Pad.forward'''

    def __init__(self, dims, stride=1):
        '''
        Args:
            dims: number of unpadding dimensions
            stride: stride length of target convolution
        '''

        super(Unpad, self).__init__()
        stride = int2tuple(stride, length=dims)
        if all(s == 1 for s in stride):
            self.subsample = nn.Sequential()
        elif dims > 3:
            raise(NotImplementedError("must have stride 1 if using >3 dims"))
        else:
            self.subsample = AvgPool(dims)(kernel_size=[1]*dims, stride=stride)

    def forward(self, x, unpad):

        return self.subsample(x[unpad])


@lru_cache(maxsize=None)
def atrous_permutation(n, k, d):
    '''computes single-dimensional atrous permutation for dilating convolutions
    Args:
        n: input size
        k: kernel size
        d: dilation rate
    Returns:
        permutation as a torch.LongTensor
    '''

    perm = torch.arange(n).roll(k//2-k).flip(0)
    for i in range(k-1, 0, -1):
        perm[i], perm[d*i] = perm[d*i].item(), perm[i].item()
    perm[:d*(k-1)+1] = perm[:d*(k-1)+1].flip(0)
    return perm.roll(-((d*(k-1)+1)//2))


class ConvPerm(nn.Module):
    '''computes permutation for convolving via FFT'''

    def __init__(self, kernel_size=1, dilation=1):
        '''
        Args:
            kernel_size: kernel size of target convolution
            dilation: dilation rate of target convolution
        '''

        super(ConvPerm, self).__init__()
        self.kernel_size = kernel_size
        self.dilation = dilation

    def forward(self, x):
        
        x = x.flip(-1).roll((self.kernel_size+1)//2, dims=-1)
        if self.dilation > 1:
            return x[...,atrous_permutation(x.shape[-1], self.kernel_size, self.dilation)]
        return x


class FNOPerm(nn.Module):
    '''computes permutation for applying FNO'''

    def __init__(self, kernel_size=1):
        '''
        Args:
            kernel_size: kernel size
        '''

        super(FNOPerm, self).__init__()
        self.kernel_size = kernel_size

    def forward(self, x):

        k = self.kernel_size
        return torch.cat([x[...,:k//2],
                          x[...,k:],
                          x[...,k//2:k]],
                         dim=-1)


class TruncateHermitian(nn.Module):
    '''applies Hermitian truncation to last dimension of the Fourier transform of a real signal'''

    def forward(self, x):

        return x[...,:x.shape[-1]//2+1]


class ExtendHermitian(nn.Module):
    '''pads truncated Hermitian signal back to full size'''

    def forward(self, x):

        n = len(x.shape) - 2
        if n == 1:
            return torch.cat([x, 
                              torch.conj(x[...,1:x.shape[-1]-1].flip(-1))],
                             dim=-1)
        # lazy hack for multi-dimensional Hermitian padding
        dim = tuple(range(-n, 0))
        return torch.fft.fftn(torch.fft.irfftn(x, dim=dim), dim=dim)


class TransposePad(nn.Module):
    '''applies padding for computing a transposed convolution via FFT'''

    def __init__(self, kernel_size=1, dilation=1):
        '''
        Args:
            kernel_size: kernel size of target transposed convolution
            dilation: dilation rate of target transposed convolution
        '''

        super(TransposePad, self).__init__()
        self.kernel_size = kernel_size
        self.dilation = dilation

    def forward(self, x):

        return F.pad(x, (0, ((x.shape[-1]-1) * (self.kernel_size-1) * self.dilation)))


@lru_cache(maxsize=None)
def transpose_permutation(n, k, d):
    '''computes single-dimensional permutation for transposed convolutions
    Args:
        n: input size
        k: kernel size
        d: dilation rate
    Returns:
        permutation as a torch.LongTensor
    '''

    perm = torch.arange(n)
    p = (k-1) * d
    for i in range((n-p) // (p+1) - 1, 0, -1):
        perm[p+i], perm[p+(p+1)*i] = perm[p+(p+1)*i].item(), perm[p+i].item()
    return perm


class TransposePerm(nn.Module):
    '''computes permutation for transposed convolving via FFT'''

    def __init__(self, kernel_size=1, dilation=1):
        '''
        Args:
            kernel_size: kernel size of target transposed convolution
            dilation: dilation rate of target transposed convolution
        '''

        super(TransposePerm, self).__init__()
        self.kernel_size = kernel_size
        self.dilation = dilation

    def forward(self, x):

        return x[...,transpose_permutation(x.shape[-1], self.kernel_size, self.dilation)]


class TransposeFlip(nn.Module):
    '''flips weights for transposed convolving via FFT'''

    def __init__(self, kernel_size=1):
        '''
        Args:
            kernel_size: kernel size of target transposed convolution
        '''

        super(TransposeFlip, self).__init__()
        self.kernel_size = kernel_size

    def forward(self, x):

        return torch.cat([x[...,:self.kernel_size].flip(-1), x[...,self.kernel_size:]], dim=-1)


class Complex2Real(nn.Module):
    '''module that returns real part of a complex input'''

    def forward(self, x):

        return x.real


def fixed(kernel_size, *args, dilation=1, stride=1, padding=0, padding_mode='circular', compact=True, odd=False, fno=False, transpose=False, **kwargs):
    '''sets fixed parameters for convolutions in XD format
    Args:
        kernel_size: kernel size of target convolution
        args: ignored
        dilation: dilation rate of target convolution
        stride: stride of target convolution
        padding: padding of target convolution
        padding_mode: padding mode of target convolution
        compact: use compact signal representation in the frequency domain
        odd: signals will have odd size in the last dimension
        fno: return parameters for FNO
        transpose: return parameter for transposed convolution
        kwargs: ignored
    Returns:
        tuple of five modules corresponding to K, L, M transforms and padding/unpadding
    '''

    dims = len(kernel_size)
    dilation = int2tuple(dilation, length=dims)
    padding = int2tuple(padding, length=dims)

    if transpose:
        if any(padding) or any((k-1)*d+1 != s for k, d, s in zip(kernel_size, dilation, int2tuple(stride, length=dims))):
            raise(NotImplementedError("transposed convolutions with stride not equal to dilated kernel size or any padding are not supported"))
        K, L, M, inpad, unpad = fixed(kernel_size, stride=1, padding=tuple((k-1)*d for k, d in zip(kernel_size, dilation)), dilation=dilation, padding_mode='zeros', compact=compact, odd=odd or (kernel_size[-1]-1)*dilation[-1] % 2)
        inpad = nn.Sequential(TensorProduct(*(TransposePad(k, d) for k, d in zip(kernel_size, dilation))), inpad)
        L = nn.Sequential(TensorProduct(*(TransposeFlip(k) for k in kernel_size)), L)
        M = nn.Sequential(TensorProduct(*(TransposePerm(k, d) for k, d in zip(kernel_size, dilation))), M)
        return K, L, M, inpad, unpad

    inpad = Pad(dims, padding=padding, padding_mode=padding_mode, dilation=dilation, kernel_size=kernel_size)
    K = Fourier(inv=True, normalized=True, dims=dims, compact=compact, odd=odd)
    if not compact:
        K = nn.Sequential(K, Complex2Real())

    if fno:
        if compact:
            L = TensorProduct(*(FNOPerm(k) for k in kernel_size[:-1]), TruncateHermitian())
        else:
            raise(NotImplementedError("must use 'compact=True' when 'fno=True'"))
    else:
        L = nn.Sequential(TensorProduct(*(ConvPerm(k, d) for k, d in zip(kernel_size, dilation))),
                          Fourier(dims=dims, compact=compact))

    M = Fourier(normalized=True, dims=dims, compact=compact)

    unpad = Unpad(dims, stride=stride)
    return K, L, M, inpad, unpad


def original(kernel_size, in_size, dilation=1, stride=1, padding=0, padding_mode='circular', compact=True, depth=1, fno=False, transpose=False, **kwargs):
    '''sets Butterfly parameters for convolutions in XD format
    Args:
        kernel_size: kernel size of target convolution
        in_size: input size
        dilation: dilation rate of target convolution
        stride: stride of target convolution
        padding: padding of target convolution
        padding_mode: padding mode of target convolution
        compact: use compact signal representation in the frequency domain
        depth: depth of Butterfly matrices
        fno: return parameters for FNO
        transpose: return parameter for transposed convolution
        kwargs: ignored
    Returns:
        tuple of five modules corresponding to K, L, M transforms and padding/unpadding
    '''

    from torch_butterfly import Butterfly
    from torch_butterfly.combine import butterfly_product
    from torch_butterfly.complex_utils import Real2Complex
    from torch_butterfly.permutation import FixedPermutation, bitreversal_permutation, perm2butterfly
    from torch_butterfly.special import fft, ifft

    dims = len(kernel_size)
    dilation = int2tuple(dilation, length=dims)
    padding = int2tuple(padding, length=dims)

    if transpose:
        if any(padding) or any((k-1)*d+1 != s for k, d, s in zip(kernel_size, dilation, int2tuple(stride, length=dims))):
            raise(NotImplementedError("transposed convolutions with stride not equal to dilated kernel size or any padding are not supported"))
        stride = 1
        padding = tuple((k-1)*d for k, d in zip(kernel_size, dilation))
        in_size = tuple(n+(n-1)*(k-1)*d for n, k, d in zip(int2tuple(in_size, length=dims), kernel_size, dilation))

    inpad = Pad(dims, in_size=in_size, padding=padding, padding_mode=padding_mode, dilation=dilation, kernel_size=kernel_size, power=True)
    in_size = inpad.in_size
    if transpose:
        inpad = nn.Sequential(TensorProduct(*(TransposePad(k, d) for k, d in zip(kernel_size, dilation))), inpad)
    if compact:
        bps = [torch.LongTensor(bitreversal_permutation(n)) for n in in_size]
    depth = int2tuple(depth, length=3)
    depth = [1+2*compact if depth[0] is None else depth[0],
             (2 if fno else 3+2*compact) if depth[1] is None else depth[1],
             1+2*compact+2*transpose if depth[2] is None else depth[2]]
    blocks = [1 if depth[0] == 2 else max(0, depth[0]-3),
              1 if depth[1] in {2, 4} else max(0, depth[1]-5),
              1 if depth[2] == 2 or (transpose and depth[2] == 4) else max(0, depth[2]-3-2*transpose)]

    kmods = []
    kmats = [ifft(n, normalized=True, with_br_perm=False) for n in in_size]
    if blocks[0]:
        kmats = [butterfly_product(Butterfly(n, n, bias=False, complex=True, init='identity', nblocks=blocks[0]), kmat)
                 for n, kmat in zip(in_size, kmats)]
    if compact:
        kmods.append(ExtendHermitian())
        if depth[0] > 2:
            kmats = [butterfly_product(perm2butterfly(bp, complex=True), kmat) for bp, kmat in zip(bps, kmats)]
        else:
            kmods.append(TensorProduct(*(FixedPermutation(bp) for bp in bps)))
    K = nn.Sequential(*kmods, TensorProduct(*kmats), Complex2Real())
        
    lmods = []
    if fno:
        if not compact:
            raise(ValueError("must use 'compact=True' when 'fno=True'"))
        if depth[1] == 1 and dims > 1:
            lmods = [TensorProduct(*(FNOPerm(k) for k in kernel_size[:-1]), nn.Identity()),
                     TensorProduct(*(Butterfly(n, n, bias=False, complex=True, init='identity') for n in in_size))]
        else:
            lmats = [perm2butterfly(FNOPerm(k)(torch.arange(n)), complex=True) for n, k in zip(in_size[:-1], kernel_size[:-1])]
            if depth[1] > 2:
                lmats = [butterfly_product(Butterfly(n, n, bias=False, complex=True, init='identity', nblocks=depth[1]-2), lmat)
                         for n, lmat in zip(in_size[:-1], lmats)]
            lmats.append(Butterfly(in_size[-1], in_size[-1], bias=False, complex=True, init='identity', nblocks=depth[1]))
            lmods = [TensorProduct(*lmats)]
    else:
        lmats = [fft(n, br_first=False, with_br_perm=False) for n in in_size]
        if blocks[1]:
            lmats = [butterfly_product(Butterfly(n, n, bias=False, complex=True, init='identity', nblocks=blocks[1]), lmat)
                     for n, lmat in zip(in_size, lmats)]
        if compact:
            if depth[1] > 4:
                lmats = [butterfly_product(lmat, perm2butterfly(bp, complex=True)) for bp, lmat in zip(bps, lmats)]
            else:
                lmods.append(TensorProduct(*(FixedPermutation(bp) for bp in bps)))
        perms = [ConvPerm(k, d) for k, d in zip(kernel_size, dilation)]
        if transpose:
            perms = [nn.Sequential(TransposeFlip(k), perm) for perm, k in zip(perms, kernel_size)]
        if depth[1] > 2:
            lmats = [butterfly_product(perm2butterfly(perm(torch.arange(n)), complex=True), lmat) 
                     for n, perm, lmat in zip(in_size, perms, lmats)]
        lmods = [TensorProduct(*lmats)] + lmods
        if depth[1] < 3:
            lmods = [TensorProduct(*perms)] + lmods
        lmods = [Real2Complex()] + lmods
    if compact:
        lmods.append(TruncateHermitian())
    L = nn.Sequential(*lmods)

    mmods = []
    mmats = [fft(n, normalized=True, br_first=False, with_br_perm=False) for n in in_size]
    if blocks[2]:
        mmats = [butterfly_product(Butterfly(n, n, bias=False, complex=True, init='identity', nblocks=blocks[2]), mmat)
                 for n, mmat in zip(in_size, mmats)]
    if compact:
        if depth[2] > 2:
            mmats = [butterfly_product(mmat, perm2butterfly(bp, complex=True)) for bp, mmat in zip(bps, mmats)]
        else:
            mmods.append(TensorProduct(*(FixedPermutation(bp) for bp in bps)))
        mmods.append(TruncateHermitian())
    if transpose:
        perms = [TransposePerm(k, d) for k, d in zip(kernel_size, dilation)]
        if depth[2] > 4:
            mmats = [butterfly_product(perm2butterfly(perm(torch.arange(n)), complex=True), mmat)
                     for n, perm, mmat in zip(in_size, perms, mmats)]
    mmods = [TensorProduct(*mmats)] + mmods
    if transpose and  depth[2] < 5:
        mmods = [TensorProduct(*perms)] + mmods
    M = nn.Sequential(Real2Complex(), *mmods)

    unpad = Unpad(dims, stride=stride)
    return K, L, M, inpad, unpad


def pad2size(x, size, **kwargs):
    '''pads input to have the given size
    Args:
        x: input tensor
        size: output size
        kwargs: passed to torch.nn.functional.pad
    Returns:
        padded tensor
    '''

    if hasattr(x, 'dim'):
        return F.pad(x,
                     sum(((n-k, 0) for n, k in zip(size, x.shape[-len(size):])), ())[::-1],
                     **kwargs)
    return x, size, kwargs


def truncate_freq(in_size, freqs):
    '''returns lower frequency slices of Fourier domain data
    Args:
        in_size: input size
        freqs: list of frequencies for each dimension
    Returns:
        slice generator
    '''

    start = [slice(None), slice(None)]
    dims = len(in_size)
    for bits in product(*[range(2)] * dims):
        if dims == 1:
            n = in_size[0]
            yield start + [slice(n-freqs[0], n) if bits[0] else slice(freqs[0])]
        else:
            yield start + [slice(n-f+bits[0], n) if b else slice(f+bits[0])
                           for b, n, f in zip(reversed(bits), in_size, freqs)]


class XD(SharedOperation):
    '''XD-Operation module for all dimensions'''

    def __init__(self, in_channels, out_channels, kernel_size, *args, weight=None, bias=None, groups=1, arch=fixed, truncate=False, dtype=torch.float32, einsum=False, **kwargs):
        '''
        Args:
            in_channels: number of input channels
            out_channels: number of output_channels
            kernel_size: kernel size as an int or tuple; passed to 'arch'
            args: passed to 'arch'
            weight: weight parameter as a torch.nn.Parameter object; also takes torch.nn.ParameterList; if None initializes using kernel_size
            bias: optional bias parameter as a torch.nn.Parameter object
            groups: number of channel groups
            arch: architecture initializer method
            truncate: truncate frequencies to kernel size
            dtype: dtype of weight tensor if initializing
            einsum: if True use torch.einsum for product
            kwargs: passed to 'arch'
        '''

        super(XD, self).__init__()
        if groups > 1:
            if in_channels % groups:
                raise(ValueError("in_channels must be divisible by groups"))
            elif in_channels != out_channels or groups != in_channels:
                raise(NotImplementedError("in_channels must equal groups"))
        self.separable = groups == out_channels > 1
        kernel_size = int2tuple(kernel_size, length=2, allow_other=True)
        self.K, self.L, self.M, self.inpad, self.unpad = arch(kernel_size, *args, **kwargs)

        if weight is None:
            self.kernel_size = kernel_size
            self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels // groups, *self.kernel_size).type(dtype))
        else:
            self.kernel_size = tuple(weight.shape[2:]) if hasattr(weight, 'shape') else tuple(max(w.shape[i] for w in weight) for i in range(2, len(weight[0].shape)))
            self.weight = weight

        self.bias = bias
        self.truncate = truncate
        self.einsum = einsum

    def forward(self, x):

        x, unpad = self.inpad(x)
        in_size = x.shape[2:]
        x = self.M(x)
        diag = self.L(pad2size(self.weight, in_size))

        if self.truncate:
            out = torch.zeros(*x.shape, dtype=x.dtype, device=x.device)
            for slices in truncate_freq(in_size, [k//2 for k in self.kernel_size[:-1]] + [self.kernel_size[-1]]):
                out[slices] = multichannel_prod(x[slices], diag[slices], separable=self.separable, einsum=self.einsum)
            x = out
        else:
            x = multichannel_prod(x, diag, separable=self.separable, einsum=self.einsum)

        x = self.K(x)
        x = self.unpad(x, unpad)
        if self.bias is None:
            return x
        return x + self.bias.reshape(1, *self.bias.shape, *[1]*len(in_size))
