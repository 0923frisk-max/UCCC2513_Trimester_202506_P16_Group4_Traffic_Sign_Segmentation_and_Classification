import os
import random
import cv2
import matplotlib.pyplot as plt
import numpy as np


def extract_color_histogram(img_bgr, bins=(8, 8, 8)):
    """Converts image to HSV space and calculates a 3D color histogram."""
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist(
        [hsv], [0, 1, 2], None, bins, [0, 180, 0, 256, 0, 256]
    )
    cv2.normalize(hist, hist)
    return hist.flatten()


def process_and_visualize(input_dir, output_dir, num_visualizations=10):
    """Processes all images, but selects 10 random images for visualization."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    all_files = [
        f
        for f in os.listdir(input_dir)
        if f.lower().endswith((".jpg", ".png", ".jpeg"))
    ]

    total_images = len(all_files)
    if total_images == 0:
        print("No valid images found in the directory.")
        return 0, 0

    # Pick 10 random images for visualization
    sample_size = min(num_visualizations, total_images)
    random_samples = set(random.sample(all_files, sample_size))

    successful_extractions = 0
    viz_count = 0

    for idx, filename in enumerate(all_files):
        img_path = os.path.join(input_dir, filename)
        img = cv2.imread(img_path)

        if img is None:
            continue

        img_resized = cv2.resize(img, (64, 64))
        color_hist_vector = extract_color_histogram(img_resized)
        successful_extractions += 1

        # Visualize only if this image was selected in the random sample
        if filename in random_samples:
            viz_count += 1
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4))

            # Original Image
            ax1.axis("off")
            ax1.imshow(cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB))
            ax1.set_title(f"Random Sample {viz_count}: {filename}")

            # Color Channel Histogram
            colors = ("b", "g", "r")
            for i, col in enumerate(colors):
                hist = cv2.calcHist([img_resized], [i], None, [256], [0, 256])
                ax2.plot(hist, color=col)

            ax2.set_xlim([0, 256])
            ax2.set_title("Color Channel Histogram")
            ax2.set_xlabel("Pixel Value")
            ax2.set_ylabel("Frequency")
            ax2.grid(True, linestyle="--", alpha=0.5)

            plt.tight_layout()
            plt.savefig(
                os.path.join(output_dir, f"random_sample_{viz_count}.png")
            )

            print(
                f"Displaying Random Image {viz_count} of {sample_size} ({filename})... Close window to continue."
            )
            plt.show()
            plt.close()

    return successful_extractions, total_images


# Main Execution
if __name__ == "__main__":
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

    test_input_directory = os.path.join(SCRIPT_DIR, "dataset", "TEST")
    visualization_output_directory = os.path.join(
        SCRIPT_DIR, "feature_visualizations"
    )

    print(f"Processing images in {test_input_directory}...")

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