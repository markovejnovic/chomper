#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from PIL import Image, ImageOps
import os
import numpy as np
from scipy.fft import fft2
from scipy.fftpack import dct
import sys

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print('compare_dft_dct.py input.img output/dir')
        exit(1)

    img = ImageOps.invert(Image.open(sys.argv[1]).convert('L'))
    img.save(os.path.join(sys.argv[2], 'input.png'))
    inputarr = np.asarray(img)
    print(f'Input shape: {inputarr.shape}')
    dftarr = fft2(inputarr)
    print(f'DFT shape: {dftarr.shape}')
    Image.fromarray(np.abs(dftarr)).convert('L').save(
        os.path.join(sys.argv[2], 'dftmagn.png'))
    Image.fromarray(np.angle(dftarr)).convert('L').save(
        os.path.join(sys.argv[2], 'dftangle.png'))
    dctarr = dct(dct(inputarr, axis=0), axis=1)
    print(f'DCT shape: {dctarr.shape}')
    Image.fromarray(np.abs(dctarr)).convert('L').save(
        os.path.join(sys.argv[2], 'dctmagn.png'))
    Image.fromarray(np.angle(dctarr)).convert('L').save(
        os.path.join(sys.argv[2], 'dctangle.png'))
