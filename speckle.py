# -*- coding: utf-8 -*-
"""
Created on Fri Apr 29 11:55:34 2022

@author: cosbo
"""

import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
import astropy.io.fits as fits
from scipy import ndimage
from PIL import Image
from photutils.detection import find_peaks
import json


if __name__ == "__main__":
    data = fits.open('speckledata.fits')[2].data
    mean = np.mean(data, axis=0)

    plt.imsave('mean.png', mean, cmap='gray')

    fourier = np.mean(np.abs(np.fft.fft2(data))**2, axis=0)
    fourier = np.fft.fftshift(fourier)

    centre = np.unravel_index(np.argmax(fourier),
                              shape=fourier.shape)
    r = 50  # да конкретно под задачу
    fr_mask = np.full_like(fourier, fill_value=False, dtype=bool)
    for i in range(fourier.shape[0]):
        for j in range(fourier.shape[1]):
            if (i-100)**2+(j-100)**2 < r**2:
                fr_mask[i, j] = True
    fr_masked = ma.MaskedArray(fourier, mask=fr_mask, dtype=np.float64)
    noise = fr_masked.mean()
    fourier = fourier - noise
    plt.imsave('fourier.png', fourier,
               vmax=np.quantile(np.mean(fourier), 0.98),
               cmap='gray')
    N = 360
    angle_mean = np.empty(N)
    rotated = np.empty((N, 200, 200))
    for i in range(N):
        rotatedim = ndimage.rotate(fourier, i * 360 / N, reshape=False)
        rotated[i] = rotatedim
    im = np.mean(rotated, axis=0)

    plt.imsave('rotaver.png', im, vmax=np.quantile(np.mean(fourier), 0.98),
               cmap='gray')

    im_masked = ma.MaskedArray(fourier/im, mask=np.invert(fr_mask)).filled(
        fill_value=0)
    binary = abs(np.fft.ifft2((im_masked), axes=(0, 1)))
    binary = np.fft.fftshift(binary)
    binary2 = Image.fromarray(binary).resize((512, 512))
    plt.imsave('binary.png', binary2,
               cmap='gray')

    '''бонуска'''
    threshold = np.mean(binary) + (2*np.std(binary))
    tbl = find_peaks(binary, threshold, box_size=11)
    tbl['peak_value'].info.format = '%.8g'
    peak_x = tbl['x_peak']
    peak_y = tbl['y_peak']
    """мы нашли ровно три пика, поэтому никаких сортировок по интенсивности
    смысла делать не имеется"""
    dis = np.sqrt((peak_x[1]-peak_x[0])**2+(peak_y[1]-peak_y[0])**2)
    dictionary = {"distance": dis*0.0206}
    with open('binary.json', 'w') as f:
        json.dump(dictionary, f)
