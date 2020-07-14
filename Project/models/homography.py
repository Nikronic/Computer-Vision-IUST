# %% import libraries

import numpy as np
import cv2
import PIL
import matplotlib.pyplot as plt

from copy import deepcopy


# %% 1 Extract Harris interest points

image1 = cv2.imread('../data/images/building1.jpg')
image2 = cv2.imread('../data/images/building2.jpg')

# 1.1 convert to grayscale
image1_gray = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY).astype(np.float32)
image2_gray = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY).astype(np.float32)

# 1.2 detect corners
image1_harris = cv2.cornerHarris(image1_gray, 5, 3, 0.02)
image2_harris = cv2.cornerHarris(image2_gray, 5, 3, 0.02)

# 1.3 dilate to emphasize features
image1_harris = cv2.dilate(image1_harris, None)
image2_harris = cv2.dilate(image2_harris, None)

# 1.4 visualization
image1[image1_harris > 0.05 * image1_harris.max()] = [0, 0, 255]
plt.imshow(image1, cmap='gray')
plt.show()

image2[image2_harris > 0.05 * image2_harris.max()] = [255, 0, 0]
plt.imshow(image2, cmap='gray')
plt.show()

# %% test
