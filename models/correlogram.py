import cv2
import numpy as np


def compute_correlogram(image, distances=[1, 3, 5], bins=64):
    """
    Compute the color correlogram for an image.

    Args:
        image (ndarray): Input image (BGR format).
        distances (list): List of pixel distances for the correlogram.
        bins (int): Number of bins for quantizing colors.

    Returns:
        ndarray: Flattened correlogram.
    """
    # Convert image to HSV color space
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    quantized_image = (hsv_image // (256 // bins)).astype(np.int32)

    height, width, _ = quantized_image.shape
    histogram = np.zeros((bins, len(distances)), dtype=np.float32)

    # Create a flattened array for easier access
    flattened_image = quantized_image[:, :, 0].flatten()
    indices = np.arange(height * width).reshape(height, width)

    for d_index, distance in enumerate(distances):
        # Shift indices to get neighbors
        shifted_indices = [
            (indices[:-distance, :], indices[distance:, :]),  # Down
            (indices[distance:, :], indices[:-distance, :]),  # Up
            (indices[:, :-distance], indices[:, distance:]),  # Right
            (indices[:, distance:], indices[:, :-distance]),  # Left
        ]

        for idx1, idx2 in shifted_indices:
            colors1 = flattened_image[idx1.flatten()]
            colors2 = flattened_image[idx2.flatten()]
            pair_counts = np.bincount(
                colors1 * bins + colors2, minlength=bins ** 2
            )
            histogram[:, d_index] += pair_counts[:bins ** 2].reshape(bins, bins).sum(axis=1)

    # Normalize the histogram
    histogram /= histogram.sum()
    return histogram.flatten()
