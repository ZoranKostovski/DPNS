import os
import numpy as np
import cv2
from tqdm import tqdm
from models.color_moments import compute_color_moments
from models.correlogram import compute_correlogram
from models.histogram import compute_histogram

# Example normalization ranges (adjust as needed)
COLOR_MOMENTS_MIN, COLOR_MOMENTS_MAX = 0, 10000
HISTOGRAM_MIN, HISTOGRAM_MAX = 0, 5
CORRELOGRAM_MIN, CORRELOGRAM_MAX = 0, 1

def normalize_and_invert(score, min_value, max_value, invert=False):
    """Normalize a score to range [0, 1] and optionally invert it."""
    normalized = (score - min_value) / (max_value - min_value)
    if invert:
        return 1 - normalized
    return normalized

def compare_images(image1, image2, method):
    """Compute similarity using the specified method."""
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

def load_images_from_directory(directory):
    """Load all images from the directory."""
    images = []
    filenames = sorted(os.listdir(directory))  # Sort to ensure consistent order
    for filename in filenames:
        if filename.endswith((".png", ".jpg", ".jpeg")):
            image_path = os.path.join(directory, filename)
            image = cv2.imread(image_path)
            if image is not None:
                images.append((filename, image))
    return images

def find_most_similar_pairs(images, models=["color_moments", "correlogram", "histogram"]):
    """Find the top 5 most similar pairs using each model."""
    similarity_results = {model: [] for model in models}

    total_combinations = len(images) * (len(images) - 1) // 2  # Total number of pairs
    progress_bar = tqdm(total=total_combinations, desc="Processing Image Pairs", unit="pair", ncols=100)

    # Compare all unique pairs of images
    for i in range(len(images)):
        for j in range(i + 1, len(images)):
            image1_name, image1 = images[i]
            image2_name, image2 = images[j]
            pair = (image1_name, image2_name)

            for model in models:
                similarity = compare_images(image1, image2, model)
                similarity_results[model].append((similarity, pair))

            progress_bar.update(1)  # Update progress bar after each pair

    progress_bar.close()  # Close the progress bar once done

    # Sort by similarity (highest first)
    top_pairs = {}
    for model, results in similarity_results.items():
        results.sort(reverse=True, key=lambda x: x[0])  # Sort by similarity score
        top_pairs[model] = results[:5]  # Take top 5 pairs

    return top_pairs

def find_combined_similarity(images, models=["color_moments", "correlogram", "histogram"]):
    """Find the top 5 most similar pairs by combining all models."""
    combined_results = []

    total_combinations = len(images) * (len(images) - 1) // 2  # Total number of pairs
    progress_bar = tqdm(total=total_combinations, desc="Processing Image Pairs (Combined)", unit="pair", ncols=100)

    # Compare all unique pairs of images
    for i in range(len(images)):
        for j in range(i + 1, len(images)):
            image1_name, image1 = images[i]
            image2_name, image2 = images[j]
            pair = (image1_name, image2_name)

            # Calculate the combined similarity for each pair
            combined_similarity = 0
            for model in models:
                similarity = compare_images(image1, image2, model)
                combined_similarity += similarity

            combined_results.append((combined_similarity, pair))
            progress_bar.update(1)  # Update progress bar after each pair

    progress_bar.close()  # Close the progress bar once done

    # Sort by combined similarity (highest first)
    combined_results.sort(reverse=True, key=lambda x: x[0])  # Sort by similarity score
    return combined_results[:5]  # Take top 5 pairs

if __name__ == "__main__":
    image_directory = "C:/Users/pc/Desktop/DPNS/data"  # Path to the 'data' directory
    images = load_images_from_directory(image_directory)

    # Find top 5 similar pairs for each model
    top_pairs_by_model = find_most_similar_pairs(images)

    # Print the results for each model
    print("Top 5 Similar Pairs by Each Model:")
    for model, pairs in top_pairs_by_model.items():
        print(f"\n{model} Top Pairs:")
        for similarity, pair in pairs:
            print(f"{pair}: Similarity = {similarity:.4f}")

    # Find top 5 combined most similar pairs
    top_combined_pairs = find_combined_similarity(images)

    # Print the combined similarity results
    print("\nTop 5 Combined Most Similar Pairs:")
    for combined_similarity, pair in top_combined_pairs:
        print(f"{pair}: Combined Similarity = {combined_similarity:.4f}")
