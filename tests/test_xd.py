import pdb
import unittest
import torch
from torch import nn
from relax.ops import Conv, ConvTranspose, FNO
from relax.xd import XD, original
from utils import TestCase


class TestConv(TestCase):

    def setUp(self):
        self.cases = []
        with torch.no_grad():
            for dims in range(1, 3):
                for in_channels, out_channels in [(1, 1), (2, 3)]:
                    for n in [15, 16]:
                        in_size = tuple(n+i for i in range(dims))
                        for k in range(2, 4):
                            kernel_size = tuple(k+i for i in range(dims))
                            for groups in [1] + ([in_channels] if in_channels == out_channels else []):
                                for dilation in range(1, 2 + 2*(k>1)):
                                    for padding in range(dilation * (k-1) + 1):
                                        mode = 'zeros' if padding > (k-1) // 2 else 'circular'
                                        for stride in range(1, k+1):

                                            conv = Conv(dims)(in_channels, 
                                                              out_channels, 
                                                              kernel_size, 
                                                              bias=False, 
                                                              groups=groups, 
                                                              dilation=dilation, 
                                                              stride=stride, 
                                                              padding=padding, 
                                                              padding_mode=mode)
                                            weight = nn.Parameter(conv.weight.data)
                                            x = torch.randn(2, in_channels, *in_size)
                                            out = conv(x)
                                            slc = [slice(None), slice(None)]
                                            if mode == 'circular':
                                                for s, o in zip(in_size, out.shape[2:]):
                                                    p = int(padding * int(bin(s)[3:]) > 0)
                                                    slc += [slice(p, o-p)]
                                            kwargs = {'in_channels': in_channels,
                                                      'out_channels': out_channels,
                                                      'n': in_size,
                                                      'k': kernel_size,
                                                      'groups': groups,
                                                      'dilation': dilation,
                                                      'stride': stride,
                                                      'padding': padding,
                                                      'padding_mode': mode}
                                            self.cases.append(([weight, x, out], slc, kwargs))

    def compare(self, weight, x, out, slc=[slice(None), slice(None)], in_channels=1, out_channels=1, n=16, k=2, **kwargs):

        xd = XD(in_channels, out_channels, k, n, weight=weight, **kwargs)
        xd.to(weight.device)
        return torch.norm(xd(x)[slc] - out[slc]) / torch.norm(out[slc])

    def test(self, cuda=False, butterfly=False):

        with torch.no_grad():
            for args, slc, kwargs in self.cases:
                n = kwargs['n'][-1]
                fno = kwargs.get('fno', False)
                if cuda:
                    for i, arg in enumerate(args):
                        args[i] = arg.cuda()

                for einsum in [False, True]:
                    kwargs['einsum'] = einsum
                    for truncate in [False] + ([True] if fno else []):
                        kwargs['truncate'] = truncate
                        for compact in [True] + ([] if fno else [False]):
                            kwargs['compact'] = compact

                            if butterfly:
                                if kwargs.get('fno', False) and int(bin(n)[3:]):
                                    continue
                                for depth in [1, None]:
                                    err = self.compare(*args, slc=slc, arch=original, depth=depth, **kwargs).item()
                                    self.assertTrue(err < 1E-5, [err, slc, kwargs])
                            else:
                                err = self.compare(*args, odd=n % 2, **kwargs).item()
                                self.assertTrue(err < 1E-6, [err, slc, kwargs])

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA unavailable")
    def test_cuda(self):

        self.test(cuda=True)

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA unavailable")
    def test_cuda_butterfly(self):

        self.test_butterfly(cuda=True)


class TestConvTranspose(TestConv):

    def setUp(self):
        self.cases = []
        with torch.no_grad():
            for dims in range(1, 3):
                for in_channels, out_channels in [(1, 1), (2, 3)]:
                    for n in [15, 16]:
                        in_size = tuple(n+i for i in range(dims))
                        for k in range(2, 4):
                            kernel_size = tuple([k]*dims)
                            for groups in [1] + ([in_channels] if in_channels == out_channels else []):
                                for dilation in range(1, 2 + 2*(k>1)):
                                    stride = (k-1) * dilation + 1
                                    padding = 0
                                    mode = 'zeros'

                                    conv = ConvTranspose(dims)(in_channels, 
                                                               out_channels, 
                                                               kernel_size, 
                                                               bias=False, 
                                                               groups=groups, 
                                                               dilation=dilation, 
                                                               stride=stride, 
                                                               padding=padding, 
                                                               padding_mode=mode)
                                    weight = nn.Parameter(conv.weight.data.transpose(0, 1))
                                    x = torch.randn(2, in_channels, *in_size)
                                    out = conv(x)
                                    slc = [slice(None), slice(None)]
                                    kwargs = {'transpose': True,
                                              'in_channels': in_channels,
                                              'out_channels': out_channels,
                                              'n': in_size,
                                              'k': kernel_size,
                                              'groups': groups,
                                              'dilation': dilation,
                                              'stride': stride,
                                              'padding': padding,
                                              'padding_mode': mode}
                                    self.cases.append(([weight, x, out], slc, kwargs))


class TestFNO(TestConv):

    def setUp(self):
        self.cases = []
        with torch.no_grad():
            for dims in range(1, 3):
                for channels in range(1, 3):
                    for n in [15, 16]:
                        for k in range(2, 4):

                            fno = FNO(channels, channels, [k]*dims)
                            weight = nn.Parameter(fno.weight.data)
                            x = torch.randn(2, channels, *[n]*dims)
                            out = fno(x)
                            slc = [slice(None), slice(None)]
                            kwargs = {'fno': True,
                                      'padding': None,
                                      'dtype': torch.complex64,
                                      'in_channels': channels,
                                      'out_channels': channels,
                                      'n': [n]*dims,
                                      'k': [2*k]*(dims-1)+[k]}
                            self.cases.append(([weight, x, out], slc, kwargs))


if __name__ == '__main__':

    unittest.main()
