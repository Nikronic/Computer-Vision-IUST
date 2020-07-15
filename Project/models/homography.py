# %% import libraries

import numpy as np
import cv2
import PIL
import matplotlib.pyplot as plt

from copy import deepcopy


# %% 1 Extract Harris interest points
def get_points(img, threshold=0.1, coordinate=False):
    """
    Extract harris points of given image

    :param img: An image of type open cv
    :param threshold: Threshold of max value in found points
    :param coordinate: Return a tuple of (x, y) coordinates instead of mask
    :return: A matrix same size as input as a binary mask of points
    """
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    img_harris = cv2.cornerHarris(img_gray, 5, 3, 0.04)
    img_points = img_harris > threshold * img_harris.max()
    if coordinate:
        img_points = (img_points * 1).nonzero()
        return img_points
    return img_points


image1 = cv2.imread('data/images/building1.jpg')
image2 = cv2.imread('data/images/building2.jpg')

image1_points = get_points(image1, coordinate=True)
image2_points = get_points(image2, coordinate=True)

# 1.4 visualization
vis = deepcopy(image1)
vis[image1_points] = [255, 0, 0]
plt.imshow(vis, cmap='gray')
plt.show()

vis = deepcopy(image2)
vis[image2_points] = [255, 0, 0]
plt.imshow(vis, cmap='gray')
plt.show()

image2[image2_harris > 0.05 * image2_harris.max()] = [255, 0, 0]
plt.imshow(image2, cmap='gray')
plt.show()

# %% test
