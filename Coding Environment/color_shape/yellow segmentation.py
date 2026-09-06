import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import math


def get_images_from_directory(path="./Inputs/Yellow Signs"):
    """Reads all supported image files from a given directory."""
    dir_path = Path(path)

    if not dir_path.exists():
        print(f"Directory not found: {path}")
        return []

    imgs = []
    # pathlib handles OS-specific slashes automatically
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}

    for item in dir_path.iterdir():
        if item.is_file() and item.suffix.lower() in valid_extensions:
            # str(item) is cleaner and safer than string concatenation
            img = cv2.imread(str(item))
            if img is not None:
                imgs.append(img)
    return imgs


def extract_and_crop_yellow_object(rgb_img):
    """Segments the largest yellow object from an RGB image."""
    # 1. Illumination Normalization (CLAHE)
    hsv = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2HSV)
    H, S, V = cv2.split(hsv)
    clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(5, 5))
    V_enhanced = clahe.apply(V)
    hsv_enhanced = cv2.merge((H, S, V_enhanced))

    # 2. Color Thresholding
    lower_yellow = np.array([10, 77, 77])
    upper_yellow = np.array([34, 255, 255])
    binary_mask = cv2.inRange(hsv_enhanced, lower_yellow, upper_yellow)

    # 3. Dynamic Morphological Operations
    dynamic_kernel_w = max(1, math.ceil(rgb_img.shape[1] * 0.02))
    dynamic_kernel_h = max(1, math.ceil(rgb_img.shape[0] * 0.02))

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (dynamic_kernel_w, dynamic_kernel_h))
    mask_closed = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
    mask_clean = cv2.morphologyEx(mask_closed, cv2.MORPH_OPEN, kernel)

    # 4. Contour Filtering
    contours, _ = cv2.findContours(mask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    final_binary_mask = np.zeros_like(mask_clean)

    if contours:
        # Keep only the largest contiguous yellow region
        largest_contour = max(contours, key=cv2.contourArea)
        cv2.drawContours(final_binary_mask, [largest_contour], -1, 255, thickness=cv2.FILLED)

    # 5. Dynamic Feathering (Gaussian Blur)
    dynamic_gaussian = math.floor(((rgb_img.shape[0] + rgb_img.shape[1]) // 2 * 0.05))

    # Ensure kernel size is odd and at least 3
    dynamic_gaussian = max(3, dynamic_gaussian if dynamic_gaussian % 2 != 0 else dynamic_gaussian + 1)

    feathered_mask = cv2.GaussianBlur(final_binary_mask, (dynamic_gaussian, dynamic_gaussian), 0)

    # 6. Apply Mask
    mask_alpha = feathered_mask.astype(float) / 255.0
    mask_alpha_3ch = cv2.merge([mask_alpha, mask_alpha, mask_alpha])
    cropped_image = (rgb_img.astype(float) * mask_alpha_3ch).astype(np.uint8)

    return cropped_image, final_binary_mask


def display_orginal_segmented_mask(rgb, output, mask):
    # Plotting
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(rgb)
    plt.title("Original")
    plt.axis("off")  # Turns off axis numbers for a cleaner look

    plt.subplot(1, 3, 2)
    plt.imshow(output)
    plt.title("Segmented Traffic Sign")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(mask, cmap="gray")
    plt.title("Yellow Mask")
    plt.axis("off")

    plt.tight_layout()
    plt.show()

import os

if __name__ == "__main__":
    dataset_path = r"C:\Users\LawYenChang\Documents\GitHub\UCCC2513_Trimester_202506_P16_Group4_Traffic_Sign_Segmentation_and_Classification\Coding Environment\dataset\TRAIN\37"
    train_set = r"\TRAIN\51"
    test_set = r"\TEST"

    image_path = os.path.join(dataset_path, train_set)

    imgs = get_images_from_directory(dataset_path)

    if not imgs:
        print("No images found to process.")

    for img in imgs:
        # Resize parameters (currently set to 1.0 multiplier)
        h, w = img.shape[:2]
        w = int(w * 1.0)
        h = int(h * 1.0)
        img = cv2.resize(img, (w, h))

        # Convert BGR (OpenCV default) to RGB (Matplotlib default)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        output, mask = extract_and_crop_yellow_object(rgb)
        display_orginal_segmented_mask(rgb, output, mask)

