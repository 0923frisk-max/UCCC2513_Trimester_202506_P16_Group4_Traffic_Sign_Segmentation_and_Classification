#coding work from Jonathan
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage import exposure
from skimage.feature import hog


def extract_color_histogram(img_bgr, bins=(8, 8, 8)):
    """Converts image to HSV space and calculates a 3D color histogram."""
    # Convert BGR to HSV for robust color representation under varying light
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    # Calculate 3D histogram across H, S, V channels
    hist = cv2.calcHist(
        [hsv], [0, 1, 2], None, bins, [0, 180, 0, 256, 0, 256]
    )

    # Normalize histogram so it is invariant to image size
    cv2.normalize(hist, hist)

    # Flatten matrix to a 1D feature vector
    return hist.flatten()


def extract_hog_features(img_bgr):
    """Resizes image to 64x64, converts to grayscale, and extracts HOG features."""
    img_resized = cv2.resize(img_bgr, (64, 64))
    gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)

    features, hog_image = hog(
        gray,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        visualize=True,
        block_norm="L2-Hys",
    )

    return features, hog_image, img_resized


def process_and_visualize(input_dir, output_dir, num_visualizations=10):
    """Extracts both Color Histogram and HOG features and plots their visual outputs."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    image_files = [
        f
        for f in os.listdir(input_dir)
        if f.lower().endswith((".jpg", ".png", ".jpeg"))
    ]

    successful_extractions = 0
    total_images = len(image_files)

    for idx, filename in enumerate(image_files):
        img_path = os.path.join(input_dir, filename)
        img = cv2.imread(img_path)

        if img is None:
            continue

        # Extract individual feature vectors
        color_hist_vector = extract_color_histogram(img)
        hog_vector, hog_image, original_resized = extract_hog_features(img)

        # Concatenate both into a single combined feature vector (for ML models)
        combined_features = np.hstack([color_hist_vector, hog_vector])

        successful_extractions += 1

        # Visualize only the specified number of samples
        if idx < num_visualizations:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4))

            # 1. Original Image
            ax1.axis("off")
            ax1.imshow(cv2.cvtColor(original_resized, cv2.COLOR_BGR2RGB))
            ax1.set_title(f"Original ({filename})")

            # 2. Color Histogram Plot (Channel breakdown)
            colors = ("b", "g", "r")
            for i, col in enumerate(colors):
                hist = cv2.calcHist([img], [i], None, [256], [0, 256])
                ax2.plot(hist, color=col)
                ax2.set_xlim([0, 256])
            ax2.set_title("Color Channel Histogram")
            ax2.set_xlabel("Pixel Value")
            ax2.set_ylabel("Frequency")
            ax2.grid(True, linestyle="--", alpha=0.5)

            plt.tight_layout()
            plt.savefig(
                os.path.join(
                    output_dir, f"feature_result_{idx+1}_{filename}.png"
                )
            )

            print(
                f"Displaying Image {idx+1} of {num_visualizations}... Close the window to continue."
            )
            plt.show()
            plt.close()

    return successful_extractions, total_images


# --- Main Execution ---
if __name__ == "__main__":
    # Get absolute directory of this script (Coding Environment)
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

    # Exact path pointing to 'Coding Environment/dataset/TEST'
    test_input_directory = os.path.join(SCRIPT_DIR, "dataset", "TEST")
    visualization_output_directory = os.path.join(
        SCRIPT_DIR, "feature_visualizations"
    )

    print(f"Processing ALL images in {test_input_directory}...")

    success_count, total = process_and_visualize(
        test_input_directory,
        visualization_output_directory,
        num_visualizations=10,
    )

    if total > 0:
        rate = (success_count / total) * 100
        print(f"\n--- Feature Extraction Results ---")
        print(f"Total Images Processed: {total}")
        print(f"Successfully Extracted: {success_count}")
        print(f"Extraction Rate: {rate:.2f}%")