import numpy as np

def compute_color_moments(image):
    moments = []
    for channel in range(image.shape[2]):  # Loop through channels
        channel_data = image[:, :, channel].flatten()
        mean = np.mean(channel_data)
        variance = np.var(channel_data)
        skewness = np.mean((channel_data - mean)**3) / (variance**1.5)
        moments.extend([mean, variance, skewness])
    return moments
