import cv2
import os
import numpy as np
from skimage.feature import hog
from skimage import exposure
import matplotlib.pyplot as plt

def extract_hog_features(image_path):
    """Reads an image, preprocesses it, and extracts HOG features."""
    img = cv2.imread(image_path)
    if img is None:
        return None, None, None
    
    img_resized = cv2.resize(img, (64, 64))
    gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    
    features, hog_image = hog(gray, orientations=9, pixels_per_cell=(8, 8),
                              cells_per_block=(2, 2), visualize=True, 
                              block_norm='L2-Hys')
    
    return features, hog_image, img_resized

def process_and_visualize(input_dir, output_dir, num_visualizations=10):
    """Processes all images, but only visualizes the first 10."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    # Grab ALL images for the true extraction rate
    image_files = [f for f in os.listdir(input_dir) if f.endswith(('.jpg', '.png'))]
    
    successful_extractions = 0
    total_images = len(image_files)

    for idx, filename in enumerate(image_files):
        img_path = os.path.join(input_dir, filename)
        features, hog_image, original_resized = extract_hog_features(img_path)
        
        if features is not None:
            successful_extractions += 1
            
            # ONLY visualize and save the first 10 images
            if idx < num_visualizations:
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)
                
                ax1.axis('off')
                ax1.imshow(cv2.cvtColor(original_resized, cv2.COLOR_BGR2RGB))
                ax1.set_title('Original Image')
                
                ax2.axis('off')
                hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))
                ax2.imshow(hog_image_rescaled, cmap=plt.cm.gray)
                ax2.set_title(f'HOG Features - Image {idx+1}')
                
                plt.savefig(os.path.join(output_dir, f'hog_result_{idx+1}.png'))
                
                print(f"Displaying Image {idx+1} of {num_visualizations}... Close the window to continue.")
                plt.show()
                plt.close()

    return successful_extractions, total_images

# --- Main Execution ---
if __name__ == "__main__":
    # Update these paths based on your folder structure
    test_input_directory = "dataset/Test"  
    visualization_output_directory = "hog_visualizations" 
    
    print(f"Processing ALL images in {test_input_directory}...")
    print("This might take a moment due to the large dataset size.")
    
    success_count, total = process_and_visualize(test_input_directory, visualization_output_directory, num_visualizations=10)
    
    if total > 0:
        extraction_rate = (success_count / total) * 100
        print(f"\n--- HOG Feature Extraction Results ---")
        print(f"Total Images Processed: {total}")
        print(f"Successfully Extracted: {success_count}")
        print(f"Extraction Rate: {extraction_rate:.2f}%")
