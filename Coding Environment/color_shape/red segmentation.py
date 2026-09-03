#Code Sample 1 [done by Jonathan Koo Hau Chung]
#below is made from Gemini's code
#define directory path of images
from pathlib import Path

input_dir=Path(r"C:/Users/jonat/OneDrive/Desktop/degree/y2s2/mini project/images/Inputs/Red signs")
#path can be changed based on the location of the images
#alternate path would be “image/Inputs/Red signs” if the images are in the same directory as the notebook
# Get a sorted list of all image paths (matches .jpg, .png, etc.)
image_paths = sorted(
    [
        p
        for p in input_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in [".jpg", 
  ".jpeg", ".png"]
    ]
)
print(f"Found {len(image_paths)} images.")


for img_path in image_paths:
    # Read the image
    img = cv2.imread(str(img_path))
   
    # Check if the image was loaded successfully
    if img is None:
        print(f"Failed to load image: {img_path}")
        continue
   
    # Convert the image from BGR to RGB (OpenCV uses BGR by default)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #then convert into HSV color space to detect red color
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)


    lower_red1 = np.array([0, 80, 60])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([168, 80, 60])
    upper_red2 = np.array([179, 255, 255])
    mask1 = cv2.inRange(img_hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(img_hsv, lower_red2, upper_red2)
    final_mask = cv2.add(mask1, mask2)


    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].imshow(img_rgb)
    axes[0].set_title("Original Image")
    axes[0].axis("off")
    axes[1].imshow(final_mask, cmap="gray")
    axes[1].axis("off")
    axes[1].set_title("Segmented Red Regions")
    plt.tight_layout()
    plt.show()



