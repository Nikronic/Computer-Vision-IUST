# %% import libraries
import numpy as np
import cv2
import random
import matplotlib.pyplot as plt

from copy import deepcopy

# %% set hyper parameters of LKT and ShiTomasi corner detector
# params for ShiTomasi corner detection
feature_params = dict(maxCorners=100,
                      qualityLevel=0.3,
                      minDistance=7,
                      blockSize=7)

# Parameters for lucas kanade optical flow
lk_params = dict(winSize=(15, 15),
                 maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# %% read video
capture = cv2.VideoCapture('data/videos/1.avi')

_, old = capture.read()
old_gray = cv2.cvtColor(old, cv2.COLOR_BGR2GRAY)
old_points = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)

_, new = capture.read()
new_gray = cv2.cvtColor(new, cv2.COLOR_BGR2GRAY)

new_points, status, error = cv2.calcOpticalFlowPyrLK(old_gray, new_gray, old_points, None, **lk_params)

old_good = old_points[status == 1]
new_good = new_points[status == 1]

homography, _ = cv2.findHomography(old_good, new_good, cv2.RANSAC, 10.0)
h, w = new.shape[:2]
vis = deepcopy(new_gray)
overlay = cv2.warpPerspective(old_gray, homography, (w, h))
vis = cv2.addWeighted(vis, 0.5, np.ones(overlay.shape, dtype=np.uint8)*255, 0.5, 0.0)

for (x0, y0), (x1, y1), good in zip(old_good[:,0], new_good[:,0], status[:,0]):
    if good:
        cv2.line(vis, (x0, y0), (x1, y1), (0, 128, 0))
    cv2.circle(vis, (x1, y1), 2, ([255, 0, 0], [0, 255, 0])[good], -1)

plt.imshow(vis, cmap='gray')
plt.show()