import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d

def kernel_gaussiano(size, sigma=1.0):
    kernel = np.fromfunction(
        lambda x, y: (1/ (2*np.pi*sigma**2)) * 
                      np.exp(-((x - (size-1)/2)**2 + (y - (size-1)/2)**2) / (2*sigma**2)),
        (size, size)
    )
    return kernel / np.sum(kernel)

def filtro_gaussiano(image, kernel):
    array = convolve2d(convolve2d(image, kernel, mode='same'), kernel.T, mode='same').astype(np.uint8)

    return array 

def grayscale(array):
    (l, c, p) = array.shape

    img_avg = np.zeros(shape=(l, c), dtype=np.uint8)
    for i in range(l):
        for j in range(c):
            r = float(array[i, j, 0])
            g = float(array[i, j, 1])
            b = float(array[i, j, 2])

            img_avg[i, j] = (r + g + b) / 3

    return img_avg