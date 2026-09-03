#Code Sample 3 [done by Law Yen Chang]
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import math

def get_Images_From_Directory(path = "./Inputs/Yellow Signs"):
    dir_path = Path(path)
    imgs = []
    for item in dir_path.iterdir():
        if item.is_file():
            if item.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                imgs.append(cv2.imread(path + "/" + item.name))
    return imgs

def extract_and_crop_yellow_object(rgb_img):
    hsv = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2HSV)
    H, S, V = cv2.split(hsv)
    clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(5, 5))
    V_enhanced = clahe.apply(V)
    hsv_enhanced = cv2.merge((H, S, V_enhanced))

    lower_yellow = np.array([10, 77, 77])   
    upper_yellow = np.array([34, 255, 255])  
    binary_mask = cv2.inRange(hsv_enhanced, lower_yellow, upper_yellow)

    dynamic_kernal_size_w = math.ceil(rgb_img.shape[1] * 0.02) 
    dynamic_kernal_size_h = math.ceil(rgb_img.shape[0] * 0.02) 

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (dynamic_kernal_size_w, dynamic_kernal_size_h))
    mask_closed = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
    
    mask_clean = cv2.morphologyEx(mask_closed, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(mask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    final_binary_mask = np.zeros_like(mask_clean)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        cv2.drawContours(final_binary_mask, [largest_contour], -1, 255, thickness=cv2.FILLED)
            
    dynamic_gaussian = math.floor(((rgb_img.shape[0] + rgb_img.shape[1]) // 2 * 0.05))

    if dynamic_gaussian % 2 == 0:
        dynamic_gaussian = dynamic_gaussian + 1
    if dynamic_gaussian < 3:
        dynamic_gaussian = 3

    feathered_mask = cv2.GaussianBlur(final_binary_mask, (dynamic_gaussian, dynamic_gaussian), 0)
    
    mask_alpha = feathered_mask.astype(float) / 255.0

    mask_alpha_3ch = cv2.merge([mask_alpha, mask_alpha, mask_alpha])

    cropped_image = (rgb_img.astype(float) * mask_alpha_3ch).astype(np.uint8)
    return cropped_image, final_binary_mask

if __name__ == "__main__":
    imgs = get_Images_From_Directory()
    rgbs = []
    for img in imgs:

        h, w = img.shape[:2]
        w = math.floor(w * 1.0) # Perform any resize for testing
        h = math.floor(h * 1.0) # Perform any resize for testing
        img = cv2.resize(img, (w, h))

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        rgbs.append(rgb)
        output, mask = extract_and_crop_yellow_object(rgb)

        plt.figure(figsize=(15, 5))
        plt.subplot(1, 3, 1)
        plt.imshow(rgb)
        plt.title("Original")
        plt.subplot(1, 3, 2)
        plt.imshow(output)
        plt.title("Segmented Traffic Sign")
        plt.subplot(1, 3, 3)
        plt.imshow(mask, cmap = "gray")
        plt.title("Yellow Mask")

        plt.tight_layout()
        plt.show()
