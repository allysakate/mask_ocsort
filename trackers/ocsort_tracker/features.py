#!/usr/bin/env python
# coding: utf-8

# Imports
import cv2 as cv
import numpy as np


# Call function ORB
def ORB():
    # Initiate ORB detector
    ORB = cv.ORB_create()

    return ORB


# Call function BRISK
def BRISK():
    # Initiate BRISK descriptor
    BRISK = cv.BRISK_create()

    return BRISK


# Call function AKAZE
def AKAZE():
    # Initiate AKAZE descriptor
    AKAZE = cv.AKAZE_create()

    return AKAZE


# Call function features
def features(descriptor, image):
    # Find the keypoints
    keypoints = descriptor.detect(image, None)

    # Compute the descriptors
    _, descriptors = descriptor.compute(image, keypoints)
    return descriptors


# Call function matcher
def matcher(descriptors1, descriptors2, matcher="BF"):

    if matcher == "BF":
        normType = cv.NORM_HAMMING

        # Create BFMatcher object
        BFMatcher = cv.BFMatcher(normType=normType, crossCheck=True)

        # Matching descriptor vectors using Brute Force Matcher
        matches = BFMatcher.match(
            queryDescriptors=descriptors1, trainDescriptors=descriptors2
        )

        # Sort them in the order of their distance
        matches = sorted(matches, key=lambda x: x.distance)
        count_match = len(matches)

    elif matcher == "FLANN":
        # FLANN parameters
        FLANN_INDEX_KDTREE = 1

        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)

        search_params = dict(checks=50)

        # Converto to float32
        descriptors1 = np.float32(descriptors1)
        descriptors2 = np.float32(descriptors2)

        # Create FLANN object
        FLANN = cv.FlannBasedMatcher(
            indexParams=index_params, searchParams=search_params
        )

        # Matching descriptor vectors using FLANN Matcher
        matches = FLANN.knnMatch(
            queryDescriptors=descriptors1, trainDescriptors=descriptors2, k=2
        )

        # Lowe's ratio test
        ratio_thresh = 0.7

        # "Good" matches
        good_matches = []

        # Filter matches
        for m, n in matches:
            if m.distance < ratio_thresh * n.distance:
                good_matches.append(m)
        count_match = len(good_matches)
    return count_match
