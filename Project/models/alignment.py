# %% import libraries
import numpy as np
import cv2
import matplotlib.pyplot as plt

from copy import deepcopy

# %% read edited image (manually)
background = cv2.imread('data/images/room.jpg', 0)
# top-left, bottom-left, bottom-right, top-right
background_points = np.array([[300, 538],
                              [385, 538],
                              [388, 689],
                              [294, 689]], dtype=np.float32)

foreground = cv2.imread('data/images/building1.jpg', 0)
foreground_points = np.array([[0, 0],
                              [foreground.shape[0], 0],
                              [foreground.shape[0], foreground.shape[1]],
                              [0, foreground.shape[1]]], dtype=np.float32)

h, mask = cv2.findHomography(foreground_points, background_points, cv2.RANSAC)
im1reg = cv2.warpPerspective(foreground, h, background.shape)

# plotting
plt.figure(figsize=(10, 10))
plt.imshow(im1reg.T+background, cmap='gray')
plt.show()

im1reg.shape
