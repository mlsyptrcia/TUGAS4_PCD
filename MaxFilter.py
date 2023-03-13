import numpy as np
import cv2

img = cv2.imread('gambar1.jpg')

kernel = np.ones((3, 3), np.uint8)

img_max = cv2.dilate(img, kernel)

cv2.imshow('Original Image', img)
cv2.imshow('Max Filter Image', img_max)
cv2.waitKey(0)
cv2.destroyAllWindows()
