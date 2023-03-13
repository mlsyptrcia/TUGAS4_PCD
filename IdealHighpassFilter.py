import cv2
import numpy as np

img = cv2.imread('gambar1.jpg')

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

d0 = 30
rows, cols = gray.shape
crow, ccol = rows//2, cols//2
mask = np.ones((rows, cols), np.uint8)
mask[crow-d0:crow+d0, ccol-d0:ccol+d0] = 0
filtered = np.fft.ifft2(np.fft.ifftshift(np.fft.fftshift(np.fft.fft2(gray)) * mask))

filtered = cv2.normalize(filtered.real, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

cv2.imshow('Original', gray)
cv2.imshow('Filtered', filtered)
cv2.waitKey(0)
cv2.destroyAllWindows()
