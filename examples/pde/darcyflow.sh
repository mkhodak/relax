#!/bin/bash

# trains FNO network on Darcy flow (resolution 85x85)
python fourier_2d.py --sub 5

# trains CNN XD network on Darcy flow (resolution 85 x 85)
python fourier_2d.py --sub 5 --arch conv --xd --arch-sgd --arch-lr 0.1 --arch-momentum 0.5
