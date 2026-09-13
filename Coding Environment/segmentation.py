import glob
import math
import os
from pathlib import Path
import cv2
import numpy as np


class IntegratedSignSegmenter:
    """Combines HSV color masking, CLAHE enhancement, and geometric shape filtering."""

    def __init__(self):
        # Broad HSV boundaries (handling low saturation / varying light)
        self.color_ranges = {
            "Blue": [(np.array([90, 60, 20]), np.array([140, 255, 255]))],
            "Yellow": [(np.array([10, 50, 50]), np.array([34, 255, 255]))],
            "Red": [
                (np.array([0, 40, 30]), np.array([10, 255, 255])),
                (np.array([160, 40, 30]), np.array([180, 255, 255])),
            ],
        }

    def _enhance_illumination(self, rgb_img: np.ndarray) -> np.ndarray:
        """Applies CLAHE on the Value channel to standardize lighting."""
        blurred = cv2.GaussianBlur(rgb_img, (3, 3), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_RGB2HSV)
        h, s, v = cv2.split(hsv)

        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(5, 5))
        v_enhanced = clahe.apply(v)
        return cv2.merge((h, s, v_enhanced))

    def _filter_by_shape(self, binary_mask: np.ndarray) -> np.ndarray:
        """Filters contours based on Area, Aspect Ratio, Solidity, and Polygon Approximations."""
        contours, _ = cv2.findContours(
            binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        final_shape_mask = np.zeros_like(binary_mask)

        valid_contours = []
        for c in contours:
            area = cv2.contourArea(c)

            # 1. Size constraint
            if 500 < area < 80000:
                x, y, w, h = cv2.boundingRect(c)
                aspect_ratio = float(w) / float(h)

                # 2. Aspect ratio constraint (traffic signs are near 1:1 ratio)
                if 0.4 <= aspect_ratio <= 2.5:
                    hull = cv2.convexHull(c)
                    hull_area = cv2.contourArea(hull)

                    # 3. Solidity constraint
                    if hull_area > 0 and (float(area) / hull_area) > 0.25:
                        valid_contours.append(c)

        if valid_contours:
            # Pick the largest valid candidate
            best_contour = max(valid_contours, key=cv2.contourArea)

            # 4. Shape geometry approximation
            epsilon = 0.035 * cv2.arcLength(best_contour, True)
            approx = cv2.approxPolyDP(best_contour, epsilon, True)
            vertices = len(approx)

            # Circular shape fitting for round signs (>= 6 vertices)
            if vertices >= 6:
                M = cv2.moments(best_contour)
                if M["m00"] != 0:
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])
                    radius = int(
                        math.sqrt(cv2.contourArea(best_contour) / math.pi)
                        * 1.02
                    )
                    cv2.circle(
                        final_shape_mask,
                        (cX, cY),
                        radius,
                        255,
                        thickness=cv2.FILLED,
                    )
            else:
                # Triangular / Rectangular polygon filling
                cv2.drawContours(
                    final_shape_mask, [approx], -1, 255, thickness=cv2.FILLED
                )

        return final_shape_mask

    def segment_sign(
        self, rgb_img: np.ndarray, color_name: str
    ) -> tuple[np.ndarray, np.ndarray]:
        """Runs unified color enhancement and shape-filtering pipeline."""
        enhanced_hsv = self._enhance_illumination(rgb_img)

        # Step 1: Color Masking
        color_mask = np.zeros(enhanced_hsv.shape[:2], dtype=np.uint8)
        ranges = self.color_ranges.get(color_name, [])

        for lower, upper in ranges:
            current_mask = cv2.inRange(enhanced_hsv, lower, upper)
            color_mask = cv2.bitwise_or(color_mask, current_mask)

        # Step 2: Morphological Cleaning
        kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11))

        opened = cv2.morphologyEx(color_mask, cv2.MORPH_OPEN, kernel_open)
        cleaned_mask = cv2.morphologyEx(
            opened, cv2.MORPH_CLOSE, kernel_close
        )

        # Step 3: Shape Filtering
        shape_mask = self._filter_by_shape(cleaned_mask)

        # Step 4: Size Check (10% boundary threshold check)
        mask_area = cv2.countNonZero(shape_mask)
        total_image_area = rgb_img.shape[0] * rgb_img.shape[1]

        if (mask_area / total_image_area) <= 0.10:
            # Fallback: Return original image and full mask when segmented area is <= 10%
            full_mask = np.full(rgb_img.shape[:2], 255, dtype=np.uint8)
            return rgb_img.copy(), full_mask

        # Step 5: Alpha Soft-Blending Output
        soft_mask = cv2.GaussianBlur(shape_mask, (3, 3), 0)
        alpha = cv2.merge([soft_mask / 255.0] * 3)
        extracted_roi = (rgb_img.astype(float) * alpha).astype(np.uint8)

        return extracted_roi, shape_mask


def run_batch_processing(
    base_dir: Path, folders: list[str], output_folder: str
):
    os.makedirs(output_folder, exist_ok=True)
    segmenter = IntegratedSignSegmenter()
    total_processed = 0

    for folder_name in folders:
        # Resolve target color category
        color_target = None
        for color in ["Red", "Blue", "Yellow"]:
            if color.lower() in folder_name.lower():
                color_target = color
                break

        if not color_target:
            continue

        target_path = base_dir / folder_name
        image_files = sorted(
            [
                p
                for p in target_path.rglob("*")
                if p.is_file()
                and p.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]
            ]
        )

        print(
            f"Processing {len(image_files)} images from '{folder_name}' [{color_target}]..."
        )

        for img_path in image_files:
            bgr_img = cv2.imread(str(img_path))
            if bgr_img is None:
                continue

            resized_bgr = cv2.resize(bgr_img, (300, 300))
            rgb_img = cv2.cvtColor(resized_bgr, cv2.COLOR_BGR2RGB)

            # Execute Combined Segmenter
            extracted_roi, shape_mask = segmenter.segment_sign(
                rgb_img, color_target
            )

            # Side-by-side comparison visualization
            roi_bgr = cv2.cvtColor(extracted_roi, cv2.COLOR_RGB2BGR)
            comparison = np.hstack((resized_bgr, roi_bgr))

            cv2.putText(
                comparison,
                "Original",
                (10, 290),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )
            cv2.putText(
                comparison,
                f"{color_target} Sign ROI",
                (310, 290),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )

            # Save result output
            out_filename = f"result_{color_target}_{img_path.name}"
            cv2.imwrite(os.path.join(output_folder, out_filename), comparison)
            total_processed += 1

    print(
        f"✅ Batch complete! Saved {total_processed} images to '{output_folder}'."
    )


if __name__ == "__main__":
    BASE_DIRECTORY = Path(
        r"C:\Users\jonat\OneDrive\Desktop\degree\y2s2\mini project\UCCC2513_Trimester_202506_P16_Group4_Traffic_Sign_Segmentation_and_Classification\Coding Environment\dataset\Demo\Inputs - Original"
    )

    TARGET_FOLDERS = ["Red signs", "Blue signs", "Yellow Signs"]
    OUTPUT_DIRECTORY = "final_color_shape_results"

    run_batch_processing(BASE_DIRECTORY, TARGET_FOLDERS, OUTPUT_DIRECTORY)