# %% import libraries

import numpy as np
import random
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
    img_harris = cv2.cornerHarris(img_gray, 7, 3, 0.03)
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


# %% Lucas-Kanade Optical Flow
def to_array(img_points):
    """
    Changes tuple of coordinates (list[x], list[y]) to a [len(x|y), 1, 2] numpy float32 array
    PS. convert type of points into 'cv2.goodFeaturesToTrack' convention.

    :param img_points: Coordinate of harris points in form of tuple
    :return: A 3D array of coordinates
    """
    assert isinstance(img_points, tuple)
    img_points_lk = np.zeros((len(img_points[0]), 1, 2), dtype=np.float32)
    img_points_lk[:, :, 0] = np.array(img_points[1]).reshape(-1, 1)
    img_points_lk[:, :, 1] = np.array(img_points[0]).reshape(-1, 1)
    return img_points_lk


def to_tuple(img_points):
    """
    Changes a [len(x|y), 1, 2] numpy float32 array coordinates to tuple of coordinates (list[x], list[y]) uint8
    PS. convert type of points into 'np.nonzero()' convention.
    :param img_points: A 3D array of coordinates
    :return: Coordinate of harris points in form of tuple
    """
    coor1 = [i[0, 0].astype(np.int) for i in img_points]
    coor2 = [i[0, 1].astype(np.int) for i in img_points]
    img_points_tuple = (coor2, coor1)
    return img_points_tuple


image1_points_lk = to_array(image1_points)
image2_points_lk = to_array(image2_points)


def lucas_kanade_tracker(img1, img2, img1_points_lk, img2_points_lk, lk_params=None, threshold=1.0):
    if lk_params is None:
        lk_params = dict(winSize=(19, 19), maxLevel=2,
                         criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    img2_points_lk, _, _ = cv2.calcOpticalFlowPyrLK(cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY),
                                                    cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY),
                                                    img1_points_lk, None, **lk_params)
    img1_points_lk_recalc, _, _ = cv2.calcOpticalFlowPyrLK(cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY),
                                                           cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY),
                                                           img2_points_lk, None, **lk_params)
    distance = abs(img1_points_lk - img1_points_lk_recalc).reshape(-1, 2).max(-1)
    status = distance < threshold

    # preserve good points
    img1_good_points = img1_points_lk[status == 1]
    img2_good_points = img2_points_lk[status == 1]
    return img1_good_points, img2_good_points


image1_good_points, image2_good_points = lucas_kanade_tracker(image1, image2, image1_points_lk, image2_points_lk)

# visualization
color = np.random.randint(0, 255, (len(image2_good_points), 3))
mask = np.zeros_like(image1)
frame = image2.copy()
for i, (i2, i1) in enumerate(zip(image2_good_points, image1_good_points)):
    a, b = i2.ravel()
    c, d = i1.ravel()
    mask = cv2.line(mask, (a, b), (c, d), color[i].tolist(), 2)
    frame = cv2.circle(frame, (a, b), 5, color[i].tolist(), -1)

vis = cv2.add(frame, mask)
plt.imshow(vis, cmap='gray')
plt.show()


# get homography
def homography(points1, points2, num_points=4):
    MIN_NUM_POINTS = 4
    assert num_points >= MIN_NUM_POINTS
    points1_indices = random.sample(list(range(len(points1))), num_points)
    points2_indices = random.sample(list(range(len(points2))), num_points)

    # build A matrix
    a_matrix = np.zeros((num_points * 2, 9))
    idx = 0
    for i, j in zip(points1_indices, points2_indices):
        a_matrix[idx, :] = np.array([-points1[i, 0, 0], -points2[i, 0, 1], -1,
                                     0, 0, 0,
                                     points2[j, 0, 0] * points1[i, 0, 0],
                                     points2[j, 0, 0] * points1[i, 0, 1],
                                     points2[j, 0, 0]])
        idx += 1
        a_matrix[idx, :] = np.array([0, 0, 0,
                                     -points1[i, 0, 0], -points1[i, 0, 1], -1,
                                     points2[j, 0, 1] * points1[i, 0, 0],
                                     points2[j, 0, 1] * points1[i, 0, 1],
                                     points2[j, 0, 1]])
        idx += 1

    u, s, v = np.linalg.svd(a_matrix)
    h_unnormalized = v[8].reshape(3, 3)
    h = (1 / h_unnormalized.flatten()[8]) * h_unnormalized
    return h


h_matrix = homography(image1_good_points, image2_good_points)

# %% test
H, status = cv2.findHomography(image1_good_points, image2_good_points, cv2.RANSAC, 10.0)
h, w = image2.shape[:2]
overlay = cv2.warpPerspective(image1, H, (w, h))
vis = cv2.addWeighted(vis, 0.5, overlay, 0.5, 0.0)

plt.imshow(vis)
plt.show()
