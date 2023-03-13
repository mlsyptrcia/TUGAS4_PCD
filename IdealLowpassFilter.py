import numpy as np
import matplotlib.pyplot as plt
from scipy import fft, ifft

def ideal_lowpass_filter(image, cutoff_frequency):
    frequency_domain = fft.fft2(image)
    freq_rows, freq_cols = frequency_domain.shape
    half_rows, half_cols = freq_rows // 2, freq_cols // 2
    row_indices = np.arange(freq_rows)
    col_indices = np.arange(freq_cols)
    col_indices[half_cols:] = col_indices[half_cols:] - freq_cols
    col_indices = np.fft.fftshift(col_indices)
    distance_matrix = np.sqrt((row_indices[:, np.newaxis] - half_rows)**2 + (col_indices[np.newaxis, :] - half_cols)**2)

    filter = np.zeros((freq_rows, freq_cols))
    filter[distance_matrix <= cutoff_frequency] = 1

    filtered_frequency_domain = frequency_domain * filter

    return np.abs(fft.ifft(filtered_frequency_domain))

from skimage import io, color

image = color.rgb2gray(io.imread('gambar1.jpg'))

filtered_image = ideal_lowpass_filter(image, 100)

plt.subplot(121), plt.imshow(image, cmap='gray'), plt.title('Gambar Asli')
plt.subplot(122), plt.imshow(filtered_image, cmap='gray'), plt.title('Hasil Filter')
plt.show()
