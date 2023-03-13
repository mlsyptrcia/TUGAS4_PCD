import numpy as np
import cv2

img = cv2.imread('gambar1.jpg')

kernel = np.ones((3, 3), np.uint8)

img_min = cv2.erode(img, kernel)

cv2.imshow('Original Image', img)
cv2.imshow('Min Filter Image', img_min)
cv2.waitKey(0)
cv2.destroyAllWindows()
