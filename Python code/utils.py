import numpy as np
import cv2
import glob
import os
from matplotlib import pyplot as plt
import math


def read_burst(burst_path, file_extension):
    # read a burst of images with extension <file_extension>
    # return a burst of images
    image_dirs = glob.glob(os.path.join(burst_path, file_extension))
    image_dirs = sorted(image_dirs)
    return np.array([cv2.imread(name) for name in image_dirs], dtype=np.uint8)


def register_burst(burst, copy=False):
    # allign a burst of images
    # source: https://learnopencv.com/image-alignment-feature-based-using-opencv-c-python/
    MAX_FEATURES = 500
    GOOD_MATCH_PERCENT = 0.15

    if copy:
        burst = np.copy(burst)

    # Convert images to grayscale
    burst_gray = np.array([cv2.cvtColor(burst[i], cv2.COLOR_BGR2GRAY)
                          for i in range(burst.shape[0])])

    # Detect ORB features and compute descriptors.
    sift = cv2.SIFT_create(MAX_FEATURES)
    keypoints = [0 for i in range(burst.shape[0])]
    descriptors = [0 for i in range(burst.shape[0])]
    for i in range(burst.shape[0]):
        keypoints[i], descriptors[i] = sift.detectAndCompute(
            burst_gray[i], None)

    (height, width) = burst[0].shape[:2]

    for i in range(1, burst.shape[0]):

        #         # Match features.
        #         # matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
        #         matcher = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)
        #         matches = list(matcher.match(descriptors[0], descriptors[i], None))

        #         # Sort matches by score
        #         matches.sort(key=lambda x: x.distance, reverse=False)

        #         # Remove not so good matches
        #         numGoodMatches = int(len(matches) * GOOD_MATCH_PERCENT)
        #         matches = matches[:numGoodMatches]

        # BFMatcher with default params
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(descriptors[0], descriptors[i], k=2)
        # Apply ratio test
        good = []
        for m, n in matches:
            if m.distance < 0.75*n.distance:
                good.append(m)
        matches = good

        # Extract location of good matches
        points1 = np.zeros((len(matches), 2), dtype=np.float32)
        points2 = np.zeros((len(matches), 2), dtype=np.float32)

        for j, match in enumerate(matches):
            points1[j, :] = keypoints[0][match.queryIdx].pt
            points2[j, :] = keypoints[i][match.trainIdx].pt

        # Find homography
        h, mask = cv2.findHomography(points1, points2, cv2.RANSAC)

        # Use homography
        burst[i] = cv2.warpPerspective(burst[i], h, (width, height))

    return burst


def get_gau_ker(ksize, sig, shape=None):
    # get gaussian blur kernel
    # return (the space domain of the blur kernel, the frequency domain of the blur kernel)
    if shape == None:
        shape = (ksize, ksize)

    # calculate kernel
    l = ksize//2
    c, r = np.meshgrid(np.linspace(-l, l, ksize), np.linspace(-l, l, ksize))
    gauss = np.exp(-(np.square(c)+np.square(r))/2/(sig**2))
    gauss /= gauss.sum()

    # pad kernel
    res = np.zeros(shape, dtype=np.float64)
    rmid = shape[0]//2
    cmid = shape[1]//2
    res[rmid-l:rmid+l+1, cmid-l:cmid+l+1] = gauss
    return res, np.abs(np.fft.fft2(res))
