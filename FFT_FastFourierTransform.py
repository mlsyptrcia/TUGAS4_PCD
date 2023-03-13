import cv2
import numpy as np

img = cv2.imread('gambar1.jpg', cv2.IMREAD_GRAYSCALE)

rows, cols = img.shape
padded_rows = cv2.getOptimalDFTSize(rows)
padded_cols = cv2.getOptimalDFTSize(cols)
padded_img = np.zeros((padded_rows, padded_cols), dtype=np.uint8)
padded_img[:rows, :cols] = img

dft = cv2.dft(np.float32(padded_img), flags=cv2.DFT_COMPLEX_OUTPUT)
dft_shift = np.fft.fftshift(dft)

magnitude_spectrum = 20 * np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]))

# Menampilkan hasil
cv2.imshow('Original Image', img)
cv2.imshow('Magnitude Spectrum', magnitude_spectrum.astype(np.uint8))
cv2.waitKey(0)
cv2.destroyAllWindows()