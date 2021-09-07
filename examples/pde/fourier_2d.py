"""
@author: Zongyi Li
This file is the Fourier Neural Operator for 2D problem such as the Darcy Flow discussed in Section 5.2 in the [paper](https://arxiv.org/pdf/2010.08895.pdf).
"""

import os
import pdb
import requests
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

import operator
from functools import reduce
from functools import partial

from timeit import default_timer

import torch.backends.cudnn as cudnn
from relax.nas import MixedOptimizer, Supernet
from relax.ops import FNO
from relax.xd import original


torch.manual_seed(0)
np.random.seed(0)


class SimpleBlock2d(nn.Module):
    def __init__(self, modes1, modes2,  width, arch='fno', einsum=True, padding_mode='circular'):
        super(SimpleBlock2d, self).__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .
        
        input: the solution of the coefficient function and locations (a(x, y), x, y)
        input shape: (batchsize, x=s, y=s, c=3)
        output: the solution 
        output shape: (batchsize, x=s, y=s, c=1)
        """

        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.fc0 = nn.Linear(3, self.width) # input channel is 3: (a(x, y), x, y)

        if arch == 'fno':
            self.conv0 = FNO(self.width, self.width, 
                [self.modes1, self.modes2], pad=True, einsum=einsum)
            self.conv1 = FNO(self.width, self.width, 
                [self.modes1, self.modes2], pad=True, einsum=einsum)
            self.conv2 = FNO(self.width, self.width, 
                [self.modes1, self.modes2], pad=True, einsum=einsum)
            self.conv3 = FNO(self.width, self.width, 
                [self.modes1, self.modes2], pad=True, einsum=einsum)
        elif arch == 'conv':
            self.conv0 = nn.Conv2d(self.width, self.width, 
                kernel_size=self.modes1 + 1, padding=6, 
                padding_mode=padding_mode, bias=False)
            self.conv1 = nn.Conv2d(self.width, self.width, 
                kernel_size=self.modes1 + 1, padding=6, 
                padding_mode=padding_mode, bias=False)
            self.conv2 = nn.Conv2d(self.width, self.width, 
                kernel_size=self.modes1 + 1, padding=6, 
                padding_mode=padding_mode, bias=False)
            self.conv3 = nn.Conv2d(self.width, self.width, 
                kernel_size=self.modes1 + 1, padding=6, 
                padding_mode=padding_mode, bias=False)
        else:
            raise NotImplementedError

        self.w0 = nn.Conv1d(self.width, self.width, 1)
        self.w1 = nn.Conv1d(self.width, self.width, 1)
        self.w2 = nn.Conv1d(self.width, self.width, 1)
        self.w3 = nn.Conv1d(self.width, self.width, 1)
        self.bn0 = torch.nn.BatchNorm2d(self.width)
        self.bn1 = torch.nn.BatchNorm2d(self.width)
        self.bn2 = torch.nn.BatchNorm2d(self.width)
        self.bn3 = torch.nn.BatchNorm2d(self.width)

        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        batchsize = x.shape[0]
        size_x, size_y = x.shape[1], x.shape[2]

        x = self.fc0(x)
        x = x.permute(0, 3, 1, 2)
        x1 = self.conv0(x)
        x2 = self.w0(x.view(batchsize, self.width, -1)).view(batchsize, self.width, size_x, size_y)
        x = self.bn0(x1 + x2)
        x = F.relu(x)
        x1 = self.conv1(x)
        x2 = self.w1(x.view(batchsize, self.width, -1)).view(batchsize, self.width, size_x, size_y)
        x = self.bn1(x1 + x2)
        x = F.relu(x)
        x1 = self.conv2(x)
        x2 = self.w2(x.view(batchsize, self.width, -1)).view(batchsize, self.width, size_x, size_y)
        x = self.bn2(x1 + x2)
        x = F.relu(x)
        x1 = self.conv3(x)
        x2 = self.w3(x.view(batchsize, self.width, -1)).view(batchsize, self.width, size_x, size_y)
        x = self.bn3(x1 + x2)


        x = x.permute(0, 2, 3, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x


class Net2d(nn.Module):
    def __init__(self, modes, width, **kwargs):
        super(Net2d, self).__init__()

        """
        A wrapper function
        """

        self.conv1 = SimpleBlock2d(modes, modes, width, **kwargs)


    def forward(self, x):
        x = self.conv1(x)
        return x.squeeze()


    def count_params(self):
        c = 0
        for p in self.parameters():
            c += reduce(operator.mul, list(p.size()))

        return c


def main(sub=5, 
         arch='fno', 
         xd=False, 
         epochs=500, 
         acc_steps=1, 
         datapath='data',
         arch_lr=0.001, 
         arch_momentum=0.0, 
         arch_sgd=False, 
         start_epoch=0):

    from utilities3 import MatReader, UnitGaussianNormalizer, LpLoss

    # Set acc_steps to 5 for full resolution
    ################################################################
    # configs
    ################################################################
    TRAIN_PATH = os.path.join(datapath, 'piececonst_r421_N1024_smooth1.mat')
    TEST_PATH = os.path.join(datapath, 'piececonst_r421_N1024_smooth2.mat')

    ntrain = 1000
    ntest = 100

    batch_size = 20 // acc_steps
    learning_rate = 0.001

    epochs = epochs
    step_size = 100
    gamma = 0.5

    modes = 12
    width = 32

    r = sub #5 # 5, 3, 2, 1
    h = int(((421 - 1)/r) + 1)
    s = h

    ################################################################
    # load data and data normalization
    ################################################################
    os.makedirs(datapath, exist_ok=True)
    if not os.path.isfile(TRAIN_PATH):
        print("downloading training data")
        with open(TRAIN_PATH, 'wb') as f:
            f.write(requests.get("https://pde-xd.s3.amazonaws.com/piececonst_r421_N1024_smooth1.mat").content)
    reader = MatReader(TRAIN_PATH)
    x_train = reader.read_field('coeff')[:ntrain,::r,::r][:,:s,:s]
    y_train = reader.read_field('sol')[:ntrain,::r,::r][:,:s,:s]

    if not os.path.isfile(TEST_PATH):
        print("downloading test data")
        with open(TEST_PATH, 'wb') as f:
            f.write(requests.get("https://pde-xd.s3.amazonaws.com/piececonst_r421_N1024_smooth2.mat").content)
    reader.load_file(TEST_PATH)
    x_test = reader.read_field('coeff')[:ntest,::r,::r][:,:s,:s]
    y_test = reader.read_field('sol')[:ntest,::r,::r][:,:s,:s]

    x_normalizer = UnitGaussianNormalizer(x_train)
    x_train = x_normalizer.encode(x_train)
    x_test = x_normalizer.encode(x_test)

    y_normalizer = UnitGaussianNormalizer(y_train)
    y_train = y_normalizer.encode(y_train)

    grids = []
    grids.append(np.linspace(0, 1, s))
    grids.append(np.linspace(0, 1, s))
    grid = np.vstack([xx.ravel() for xx in np.meshgrid(*grids)]).T
    grid = grid.reshape(1,s,s,2)
    grid = torch.tensor(grid, dtype=torch.float)
    x_train = torch.cat([x_train.reshape(ntrain,s,s,1), grid.repeat(ntrain,1,1,1)], dim=3)
    x_test = torch.cat([x_test.reshape(ntest,s,s,1), grid.repeat(ntest,1,1,1)], dim=3)

    train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test, y_test), batch_size=batch_size, shuffle=False)

    ################################################################
    # training and evaluation
    ################################################################
    model = Net2d(modes, width, arch=arch, einsum=True)

    # Convert to NAS search space, if applicable
    if xd:
        if arch == 'fno':
            raise(NotImplementedError("XD substitution not implemented for FNO"))
        X = torch.zeros([batch_size, s, s, 3])
        Supernet.create(model, in_place=True)
        named_modules = []
        for name, layer in model.named_modules():
            if isinstance(layer, torch.nn.Conv2d):
                named_modules.append((name, layer))
        # Only patch conv2d
        model.conv2xd(X[:1], named_modules=named_modules, arch=original, depth=1, compact=False, verbose=True)
    else:
        arch_lr = 0.0

    cudnn.benchmark = True
    model.cuda()
    print(model)
    print(model.count_params())

    if xd:
        momentum = partial(torch.optim.SGD, momentum=arch_momentum)
        arch_opt = momentum if arch_sgd else torch.optim.Adam
        opts = [torch.optim.Adam(model.model_weights(), lr=learning_rate, weight_decay=1E-4),
                arch_opt([{'params': list(model.arch_params())}],
                         lr=arch_lr, weight_decay=1e-4)]
    else:
        opts = [torch.optim.Adam(model.parameters(), 
            lr=learning_rate, weight_decay=1e-4)]
    optimizer = MixedOptimizer(opts)


    def weight_sched(epoch):

        return gamma ** (epoch // step_size)

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=weight_sched, last_epoch=start_epoch-1)

    #weight_sched = torch.optim.lr_scheduler.StepLR(optimizer, 
    #    step_size=step_size, gamma=gamma)
    #optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)

    myloss = LpLoss(size_average=False)
    y_normalizer.cuda()
    for ep in range(epochs):
        model.train()
        t1 = default_timer()
        train_mse = 0
        optimizer.zero_grad()
        for i, (x, y) in enumerate(train_loader):
            x, y = x.cuda(), y.cuda()
            
            # loss = F.mse_loss(model(x).view(-1), y.view(-1), reduction='mean')
            out = model(x)
            out = y_normalizer.decode(out)
            y = y_normalizer.decode(y)
            loss = myloss(out.view(batch_size,-1), y.view(batch_size,-1))
            loss.backward()

            if (i + 1) % acc_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

            # Is this accumulating too many times?
            train_mse += loss.item()

        scheduler.step()

        model.eval()
        abs_err = 0.0
        rel_err = 0.0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.cuda(), y.cuda()

                out = model(x)
                out = y_normalizer.decode(model(x))

                rel_err += myloss(out.view(batch_size,-1), y.view(batch_size,-1)).item()

        train_mse/= ntrain
        abs_err /= ntest
        rel_err /= ntest

        t2 = default_timer()
        print(ep, t2-t1, train_mse, rel_err)

if __name__ == '__main__':

    import fire

    fire.Fire(main)
