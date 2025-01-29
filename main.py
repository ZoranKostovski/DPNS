import cv2
import numpy as np

from models.color_moments import compute_color_moments
from models.correlogram import compute_correlogram
from models.histogram import compute_histogram


COLOR_MOMENTS_MIN, COLOR_MOMENTS_MAX = 0, 10000
HISTOGRAM_MIN, HISTOGRAM_MAX = 0, 5
CORRELOGRAM_MIN, CORRELOGRAM_MAX = 0, 1

def normalize_and_invert(score, min_value, max_value, invert=False):
    normalized = (score - min_value) / (max_value - min_value)
    if invert:
        return 1 - normalized
    return normalized

import numpy as np

def compare_images(image1, image2, method):
    if method == "color_moments":
        desc1 = np.array(compute_color_moments(image1))  # Convert to NumPy array
        desc2 = np.array(compute_color_moments(image2))  # Convert to NumPy array
        distance = np.linalg.norm(desc1 - desc2)
        return normalize_and_invert(distance, COLOR_MOMENTS_MIN, COLOR_MOMENTS_MAX, invert=True)
    elif method == "correlogram":
        desc1 = np.array(compute_correlogram(image1))  # Convert to NumPy array
        desc2 = np.array(compute_correlogram(image2))  # Convert to NumPy array
        similarity = np.dot(desc1, desc2) / (np.linalg.norm(desc1) * np.linalg.norm(desc2))
        return normalize_and_invert(similarity, CORRELOGRAM_MIN, CORRELOGRAM_MAX)
    elif method == "histogram":
        desc1 = np.array(compute_histogram(image1))  # Convert to NumPy array
        desc2 = np.array(compute_histogram(image2))  # Convert to NumPy array
        distance = np.linalg.norm(desc1 - desc2)
        return normalize_and_invert(distance, HISTOGRAM_MIN, HISTOGRAM_MAX, invert=True)


if __name__ == "__main__":
    # Use absolute paths for images
    image1_path = "C:/Users/pc/Desktop/DPNS/data/0001.png"
    image2_path = "C:/Users/pc/Desktop/DPNS/data/0002.png"

    # Load images
    image1 = cv2.imread(image1_path)
    image2 = cv2.imread(image2_path)

    # Validate images
    if image1 is None or image2 is None:
        print("Error: One or both image files could not be loaded. Check the file paths.")
        exit(1)

    # Compare images
    for method in ["color_moments", "correlogram", "histogram"]:
        similarity = compare_images(image1, image2, method)
        print(f"{method} similarity: {similarity:.4f}")
