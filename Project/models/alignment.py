# %% import libraries
import numpy as np
import cv2
import matplotlib.pyplot as plt

from copy import deepcopy

# %% read edited image (manually)
from docutils.nodes import footnote_reference

background = cv2.imread('data/images/room.jpg', 0)
# bottom-left, top-left, bottom-right, top-right
background_points = np.array([[385, 538],
                              [300, 538],
                              [388, 689],
                              [294, 689]], dtype=np.float32)
background_points = np.roll(background_points, 0, axis=1)

foreground = cv2.imread('data/images/building2.jpg', 0)
foreground_points = np.array([[foreground.shape[0], 0],
                              [0, 0],
                              [foreground.shape[0], foreground.shape[1]],
                              [0, foreground.shape[1]]], dtype=np.float32)
foreground_points = np.roll(foreground_points, 1, axis=1)

h, mask = cv2.findHomography(foreground_points, background_points, cv2.RANSAC)
im1reg = cv2.warpPerspective(foreground, h, background.shape)

# plotting
plt.figure(figsize=(10, 10))
plt.imshow(im1reg.T+background, cmap='gray')
plt.show()

im1reg.shape
