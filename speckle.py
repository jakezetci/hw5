# -*- coding: utf-8 -*-
"""
Created on Fri Apr 29 11:55:34 2022

@author: cosbo
"""

import numpy as np
import numpy.ma as ma
import matplotlib
import matplotlib.pyplot as plt
import astropy.io.fits as fits
from scipy import ndimage

if __name__ == "__main__":

    data = fits.open('speckledata.fits')[2].data
    mean = np.mean(data, axis=0)
    plt.figure()
    a = plt.imsave('mean.png',mean)  
    

    fourier = np.abs(np.fft.fft(data, n=1, axis=0)[0])**2


    centre = np.unravel_index(np.argmax(fourier),
                              shape=fourier.shape)
    r = 50  # да конкретно под задачу да говнокод
    fr_mask = np.full_like(fourier, fill_value=False, dtype=bool)
    fr_mask[centre[0]-r:centre[0]+r, centre[1]-r:centre[1]+r] = True

    fr_masked = ma.MaskedArray(fourier, mask=fr_mask, dtype=np.float64)
    noise = fr_masked.mean()
    fourier = fourier - noise

    N = 360
    angle_mean = np.empty(N)
    rotated = np.empty((N, 200, 200))
    for i in range(N):
        rotatedim = ndimage.rotate(fourier, i * 360 / N, reshape=False)
        angle_mean[i] = np.nanmean(np.mean(rotatedim, axis=0))
        rotated[i] = ndimage.rotate(rotatedim/angle_mean[i], -i * 360 / N,
                                    reshape=False)
    im = np.mean(rotated, axis=0)

    plt.figure()
    b = plt.imsave('fourier.png',im,
                   cmap='gray')


    im_masked = np.atleast_3d(ma.MaskedArray(im, mask=fr_mask).filled(0))
    im_new = np.fft.irfft(im_masked, n=101)
    plt.figure()
    plt.imshow(np.mean(im_new, axis=-1), cmap='gray')
