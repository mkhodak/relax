import unittest
try:
    import torch_butterfly
    BUTTERFLY = True
except ImportError:
    BUTTERFLY = False 


class TestCase(unittest.TestCase):

    def test(self, butterfly=False):

        pass

    @unittest.skipIf(not BUTTERFLY, "torch_butterfly not found")
    def test_butterfly(self, **kwargs):

        self.test(butterfly=True, **kwargs)
