import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('gambar1.jpg', 0)

dft = np.fft.fft2(img)

dft_shift = np.fft.fftshift(dft)

radius = 50

rows, cols = img.shape
crow, ccol = rows//2, cols//2
u, v = np.meshgrid(np.arange(cols), np.arange(rows))
d_uv = np.sqrt((u - ccol)**2 + (v - crow)**2)
H = 1 - np.exp(-1*(d_uv**2)/(2*(radius**2)))

dft_shift_filtered = dft_shift * H

dft_filtered = np.fft.ifftshift(dft_shift_filtered)

img_filtered = np.fft.ifft2(dft_filtered)
img_filtered = np.abs(img_filtered)

plt.subplot(121), plt.imshow(img, cmap='gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(img_filtered, cmap='gray')
plt.title('Gaussian Highpass Filtered Image'), plt.xticks([]), plt.yticks([])
plt.show()