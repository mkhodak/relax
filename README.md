# relax

Simplified and packaged implementation of XD-operations from the paper [Rethinking Neural Operations for Diverse Tasks](https://arxiv.org/abs/2103.15798).
Shell scripts in <tt>examples/resnet</tt> and <tt>examples/pde</tt> reproduce the permuted CIFAR and Darcy flow results from the paper.
Install instructions below; a Dockerfile is also provided.
```
conda install pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch -c nvidia
pip install fire h5py scipy tensorboard tensorboardX requests
git clone --single-branch --branch pt1.8 https://github.com/HazyResearch/butterfly.git butterfly
cd butterfly && export FORCE_CUDA="1" && python setup.py install
pip install -e .
```

## usage

XD is straightforward to apply by inserting code similar to the following into a PyTorch training script. The code takes an existing CNN model, re-parameterizes all of its convolutions using the XD relaxation, and defines an optimizer that updates the original model weights using SGD and the architecture parameters (K-matrices) using Adam. More detailed examples, including how to handle learning rate schedules and how to re-parameterize only specific convolutions, can be found in the 'examples' folder.

```
### script setup
# 'model' is a standard PyTorch convolutional neural network
# 'X' is a Tensor of the same size as a batch of input data
###

from torch import nn
from relax.nas import MixedOptimizer, Supernet
from relax.xd import original

model = Supernet.create(model)
model.conv2xd(X[:1], arch=original, verbose=True)
optimizer = MixedOptimizer([
                            nn.optim.SGD(model.model_weights(), lr=1E-1), # suggested to use same optimizer as original model
                            nn.optim.Adam(model.arch_params(), lr=1E-4), # tune the architecture learning rate as needed
                            ])
                            
### continue script        
```
