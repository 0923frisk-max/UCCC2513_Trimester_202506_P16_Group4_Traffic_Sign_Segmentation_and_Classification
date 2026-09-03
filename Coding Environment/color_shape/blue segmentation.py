#Code Sample 2 [done by Christopher Yong Wen Jie]
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

class BlueSignSegmenter:
    def __init__(self):
       
        self.lower_blue = np.array([100, 120, 40])
        self.upper_blue = np.array([135, 255, 255])
        
        self.kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        self.kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
        
    def _enhance_illumination(self, rgb_image):
        blurred = cv2.GaussianBlur(rgb_image, (3, 3), 0)
        hsv_image = cv2.cvtColor(blurred, cv2.COLOR_RGB2HSV)
        
        h, s, v = cv2.split(hsv_image)
        clahe_processor = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(5, 5))
        v_equalized = clahe_processor.apply(v)
        
        return cv2.merge((h, s, v_equalized))

    def _generate_clean_mask(self, hsv_image):
      
        raw_mask = cv2.inRange(hsv_image, self.lower_blue, self.upper_blue)
        
        opened_mask = cv2.morphologyEx(raw_mask, cv2.MORPH_OPEN, self.kernel_open)

        closed_mask = cv2.morphologyEx(opened_mask, cv2.MORPH_CLOSE, self.kernel_close)

        refined_mask = np.zeros_like(closed_mask)
        contours, _ = cv2.findContours(closed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            valid_contours = [c for c in contours if cv2.contourArea(c) > 300]
            if valid_contours:
                largest_blob = max(valid_contours, key=cv2.contourArea)
                cv2.drawContours(refined_mask, [largest_blob], -1, 255, thickness=cv2.FILLED)
            
        return refined_mask

    def isolate_sign(self, rgb_image):
        enhanced_hsv = self._enhance_illumination(rgb_image)
        
        base_mask = self._generate_clean_mask(enhanced_hsv)
        
        soft_mask = cv2.GaussianBlur(base_mask, (3, 3), 0)
        alpha_channel = soft_mask.astype(float) / 255.0
        alpha_3d = cv2.merge([alpha_channel, alpha_channel, alpha_channel])
        
        # Step D: Final Extraction
        extracted_roi = (rgb_image.astype(float) * alpha_3d).astype(np.uint8)
        
        return extracted_roi, base_mask

def execute_blue_pipeline(target_directory):
    folder_path = Path(target_directory)
    
    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp"}
    dataset_paths = sorted([
        p for p in folder_path.rglob("*") 
        if p.is_file() and p.suffix.lower() in valid_extensions
    ])
    
    print(f"Pipeline Initialized: {len(dataset_paths)} images located in '{folder_path.name}'.")

    segmenter = BlueSignSegmenter()

    for img_path in dataset_paths:
        raw_bgr = cv2.imread(str(img_path))
        if raw_bgr is None:
            print(f"Warning: Unreadable file skipped -> {img_path.name}")
            continue
            
        standardized_bgr = cv2.resize(raw_bgr, (300, 300))
        input_rgb = cv2.cvtColor(standardized_bgr, cv2.COLOR_BGR2RGB)
        
        final_output, final_mask = segmenter.isolate_sign(input_rgb)
  
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        axes[0].imshow(input_rgb)
        axes[0].set_title("Source Image")
        axes[0].axis("off")
        
        axes[1].imshow(final_output)
        axes[1].set_title("Extracted Region of Interest")
        axes[1].axis("off")
        
        axes[2].imshow(final_mask, cmap="gray")
        axes[2].set_title("Binary Segmentation Mask")
        axes[2].axis("off")

        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    # Point this to your specific blue signs folder
    INPUT_DATA_PATH = "Blue Signs"
    
    # Trigger the pipeline
    execute_blue_pipeline(INPUT_DATA_PATH)


