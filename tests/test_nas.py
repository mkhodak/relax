import pdb
import unittest
import torch
from relax.nas import Supernet
from relax.xd import original
from fourier_2d import Net2d
from resnet import resnet20
from utils import TestCase


class TestResNet20(TestCase):

    def setUp(self):

        self.model = resnet20()
        self.X = torch.randn(2, 3, 32, 32)
        self.out = self.model(self.X)
        self.weights = sum(p.numel() for p in self.model.parameters())
        self.names = {n for n, p in self.model.named_modules()}

    def compare(self, model):

        return torch.norm(self.out - model(self.X)) / torch.norm(self.out)

    def test(self, butterfly=False):

        for einsum in [False, True]:
            for compact in [False, True]:
                kwargs = {'einsum': einsum, 'compact': compact}
                if butterfly:
                    params = {}
                    for depth in [1, None]:
                        model = Supernet.create(self.model)
                        nm = [(n, m) for n, m in model.named_modules() if n in self.names]
                        model.conv2xd(self.X, 
                                      named_modules=nm,
                                      arch=original, depth=depth, **kwargs)
                        self.assertEqual(self.weights,
                                         sum(p.numel() for p in model.model_weights()),
                                         "different number of model weights")
                        params[depth] = sum(p.numel() for p in model.arch_params())
                        err = self.compare(model).item()
                        self.assertTrue(err < 1E-5, (err, einsum, compact, butterfly))
                    self.assertTrue(params[None] > params[1] > 0, 
                                    "incorrect number of architecture parameters")
                else:
                    model = Supernet.create(self.model)
                    nm = [(n, m) for n, m in model.named_modules() if n in self.names]
                    model.conv2xd(self.X, named_modules=nm, **kwargs)
                    self.assertEqual(self.weights,
                                     sum(p.numel() for p in model.model_weights()),
                                     "different number of model weights")
                    self.assertEqual(0, sum(p.numel() for p in model.arch_params()),
                                     "nonzero number of architecture parameters")
                    err = self.compare(model).item()
                    self.assertTrue(err < 1E-5, (err, einsum, compact, butterfly))


class TestNet2d(TestResNet20):

    def setUp(self):

        self.model = Net2d(12, 32, arch='conv', padding_mode='zeros')
        self.X = torch.randn(2, 85, 85, 3)
        self.out = self.model(self.X)
        self.weights = sum(p.numel() for p in self.model.parameters())
        self.names = {n for n, m in self.model.named_modules() if '.' in n and 'conv' in n.split('.')[-1]}


if __name__ == '__main__':

    unittest.main()
