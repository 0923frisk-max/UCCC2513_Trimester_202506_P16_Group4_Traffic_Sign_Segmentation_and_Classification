# Import the specific functions you need from the (renamed) file
from color_shape.yellow_segmentation import get_images_from_directory, extract_and_crop_yellow_object

# Load the list of images
train_images = get_images_from_directory(r"./dataset/TRAIN/45")

# Iterate through the list and process each image individually
for img in train_images:
    cropped_img, mask = extract_and_crop_yellow_object(img)
    # Add your displaying or saving logic here