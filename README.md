# relax

Simplified and packaged implementation of XD-operations from the paper [Rethinking Neural Operations for Diverse Tasks](https://arxiv.org/abs/2103.15798).
Shell scripts in <tt>examples/resnet</tt> and <tt>examples/pde</tt> reproduce the permuted CIFAR and Darcy flow results from the paper.
Install instructions below; a Dockerfile is also provided.
```
conda install pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch -c nvidia
pip install fire h5py scipy tensorboard tensorboardX
git clone --single-branch --branch pt1.8 https://github.com/HazyResearch/butterfly.git butterfly
cd butterfly && export FORCE_CUDA="1" && python setup.py install
pip install -e .
```
