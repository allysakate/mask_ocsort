import numpy as np
import cv2
from trackers.ocsort_tracker.features import features, matcher, AKAZE, ORB, BRISK


def _matcher(descriptor, a, b, match="FLANN"):
    a = a.reshape(256, 256)
    b = b.reshape(256, 256)
    descriptor1 = features(descriptor, a)
    descriptor2 = features(descriptor, b)
    num_match = matcher(descriptor1, descriptor2, match)
    print(num_match)
    # if descriptor1 is None or descriptor2 is  None:
    #     num_match = 0
    # else:
    #     num_match = matcher(descriptor1, descriptor2, match)
    return num_match


for desc in ["AKAZE", "BRISK", "ORB"]:
    for match in ["BF", "FLANN"]:
        print(f"desc: {desc}, match: {match}")
        if desc == "AKAZE":
            descriptor = AKAZE()
        elif desc == "ORB":
            descriptor = ORB()
        elif desc == "BRISK":
            descriptor = BRISK()
        match_dict = {}
        # det_imgs = ["cropped/100_1.jpg", "cropped/100_2.jpg", "cropped/100_3.jpg", "cropped/100_4.jpg", "cropped/100_5.jpg"]
        # trk_imgs = ["75_1.jpg", "cropped/75_2.jpg", "cropped/75_3.jpg"] #, "cropped/25_4.jpg", "cropped/25_5.jpg"]
        det_imgs = [
            "cropped/50_1.jpg",
            "cropped/50_2.jpg",
            "cropped/50_3.jpg",
            "cropped/50_4.jpg",
            "cropped/50_5.jpg",
        ]
        trk_imgs = [
            "cropped/25_1.jpg",
            "cropped/25_2.jpg",
            "cropped/25_3.jpg",
            "cropped/25_4.jpg",
            "cropped/25_5.jpg",
        ]
        fm_matrix = np.zeros((len(det_imgs), len(trk_imgs)))
        for d_idx, det_img in enumerate(det_imgs):
            crop = cv2.imread(det_img)
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            print(crop.shape, gray.shape)
            det_img = np.array(gray).flatten()
            match_dict = {}
            for t_idx, trk_img in enumerate(trk_imgs):
                crop = cv2.imread(trk_img)
                gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                trk_img = np.array(gray).flatten()
                fm_matrix[d_idx, t_idx] = _matcher(descriptor, det_img, trk_img, match)
        max_v = np.amax(fm_matrix)
        max_m = fm_matrix.argmax(axis=1)
        print(max_m, max_v)
        print(fm_matrix / max_v)
        print()
