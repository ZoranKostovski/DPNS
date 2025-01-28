import cv2
import numpy as np

def compute_histogram(image, bins=32, color_space='RGB'):
    if color_space == 'HSV':
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist = []
    for channel in range(image.shape[2]):
        hist_channel = cv2.calcHist([image], [channel], None, [bins], [0, 256])
        hist.append(cv2.normalize(hist_channel, hist_channel).flatten())
    return np.concatenate(hist)
