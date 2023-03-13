import cv2
import numpy as np

img = cv2.imread('gambar1.jpg')

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

filter_size = 11
sigma = 5
kernel = cv2.getGaussianKernel(filter_size, sigma)

filtered = cv2.filter2D(gray, -1, kernel)

cv2.imshow('Original Image', gray)
cv2.imshow('Filtered Image', filtered)
cv2.waitKey(0)
cv2.destroyAllWindows()