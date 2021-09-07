#!/bin/bash

# trains vanilla ResNet-20 on permuted (bit-reversed) CIFAR-10
python trainer.py --permute --save-dir results/permute/conv

# trains ResNet-20 XD on permuted (bit-reversed) CIFAR-10
python trainer.py --xd --permute --save-dir results/permute/xd --arch-adam --arch-lr 1E-3 --kmatrix-depth 3
