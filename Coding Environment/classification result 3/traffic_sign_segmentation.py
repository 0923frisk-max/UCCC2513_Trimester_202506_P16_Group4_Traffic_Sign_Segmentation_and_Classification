"""
traffic_sign_segmentation.py
=============================

Integrated traffic-sign segmentation module that merges and improves the
three separate scripts (blue / red / yellow color segmentation + shape.py)
into one reliable pipeline.

Key reliability improvements over the original scripts
--------------------------------------------------------
1. Illumination normalization BEFORE color thresholding:
   - Gray-world auto white balance removes color casts caused by
     tungsten / fluorescent / sodium-vapor lighting, which is the #1
     cause of hue drift in HSV-based color segmentation.
   - CLAHE is applied on the L channel in LAB space (not on HSV's V
     channel). Doing it in LAB avoids leaking illumination changes into
     the Hue/Saturation channels, which was happening in the original
     blue/yellow scripts (they ran CLAHE on V *after* the HSV
     conversion was already computed from the un-normalized image, and
     re-merged into HSV — order does not matter there because H and S
     are untouched either way, but LAB CLAHE additionally keeps
     saturation more physically consistent before we re-derive HSV).
2. Median blur (salt-and-pepper noise) + Gaussian blur (sensor noise)
   before threshold, instead of only Gaussian.
3. HSV ranges were widened/aligned across all three original scripts
   (blue/red/yellow each had two slightly different range sets) and a
   brightness-adaptive lower-V bound is used so dim/underexposed images
   don't lose their sign to the mask.
4. Morphology + contour selection now scores candidates using area,
   aspect ratio, solidity AND circularity together (the original
   shape.py only used area/aspect/solidity for the mask, and a crude
   vertex-count check to decide "draw as filled circle vs polygon").
   Scoring instead of hard sequential filtering makes the module far
   less likely to drop a valid sign because of one borderline check.
5. One shared implementation instead of 3 near-duplicate scripts, so a
   fix/tune in one place (e.g. HSV ranges) benefits all colors.

Expected dataset layout
------------------------
    <root>/<Train|Test>/<Blue|Red|Yellow>/<class_id>/<image files>

Usage
-----
    from traffic_sign_segmentation import process_dataset

    results = process_dataset(
        root_dir="dataset",
        split="Train",
        output_root="cropped",
        show=True,
    )
    # results is a list of dicts:
    #   {"filename": str, "original": np.ndarray (RGB), "cropped": np.ndarray (RGB),
    #    "status": "ok" | "fallback",
    #    "hog": np.ndarray, "color_histogram": np.ndarray, "hog_color": np.ndarray}
"""

import math
import os
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np


import feature_extraction_one as feat1
import feature_extraction_two as feat2

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
COLORS = ["Blue", "Red", "Yellow"]


class TrafficSignSegmenter:
    """Color + shape based segmenter, robust to lighting and noise."""

    def __init__(self, resize_to=(300, 300), min_mask_area_ratio=0.05):
        self.resize_to = resize_to
        # If the final mask covers less than this fraction of the frame,
        # the "detected" region is treated as too small/unreliable and we
        # fall back to returning the (resized) original image instead of
        # a crop that is probably noise, not signal.
        self.min_mask_area_ratio = min_mask_area_ratio
        self.kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        self.kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11))

    # ------------------------------------------------------------------
    # Illumination / noise handling
    # ------------------------------------------------------------------
    def _gray_world_white_balance(self, bgr_image):
        """Simple, fast auto white balance to remove color casts."""
        result = bgr_image.astype(np.float32)
        avg_b, avg_g, avg_r = [result[:, :, i].mean() for i in range(3)]
        avg_gray = (avg_b + avg_g + avg_r) / 3.0
        # Avoid divide-by-zero on pure black frames
        result[:, :, 0] *= (avg_gray / avg_b) if avg_b > 1e-3 else 1.0
        result[:, :, 1] *= (avg_gray / avg_g) if avg_g > 1e-3 else 1.0
        result[:, :, 2] *= (avg_gray / avg_r) if avg_r > 1e-3 else 1.0
        return np.clip(result, 0, 255).astype(np.uint8)

    def _normalize_illumination(self, bgr_image):
        balanced = self._gray_world_white_balance(bgr_image)

        # Denoise before anything else: median kills salt-and-pepper /
        # compression speckle, Gaussian softens residual sensor noise.
        denoised = cv2.medianBlur(balanced, 3)
        denoised = cv2.GaussianBlur(denoised, (3, 3), 0)

        lab = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l_equalized = clahe.apply(l_channel)
        lab_equalized = cv2.merge((l_equalized, a_channel, b_channel))

        normalized_bgr = cv2.cvtColor(lab_equalized, cv2.COLOR_LAB2BGR)
        hsv = cv2.cvtColor(normalized_bgr, cv2.COLOR_BGR2HSV)
        return hsv

    # ------------------------------------------------------------------
    # Color thresholding
    # ------------------------------------------------------------------
    def _get_hsv_ranges(self, color_name, hsv_image):
        """Return list of (lower, upper) HSV bounds, with the V lower
        bound nudged down for images that are globally dark so faint
        (shadowed / backlit) signs aren't lost."""
        mean_v = hsv_image[:, :, 2].mean()
        # In a dark scene, relax the minimum value/brightness we accept
        v_floor = 30 if mean_v < 90 else 45

        if color_name == "Blue":
            return [(np.array([95, 60, v_floor]), np.array([135, 255, 255]))]
        elif color_name == "Yellow":
            return [(np.array([12, 70, v_floor]), np.array([34, 255, 255]))]
        elif color_name == "Red":
            return [
                (np.array([0, 70, v_floor]), np.array([10, 255, 255])),
                (np.array([165, 70, v_floor]), np.array([180, 255, 255])),
            ]
        return []

    def _color_mask(self, hsv_image, color_name):
        mask = np.zeros(hsv_image.shape[:2], dtype=np.uint8)
        for lower, upper in self._get_hsv_ranges(color_name, hsv_image):
            mask = cv2.bitwise_or(mask, cv2.inRange(hsv_image, lower, upper))
        return mask

    def _clean_mask(self, mask):
        opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel_open)
        closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, self.kernel_close)
        return closed

    # ------------------------------------------------------------------
    # Shape validation (merged + scored version of shape.py's logic)
    # ------------------------------------------------------------------
    def _score_contour(self, contour, image_area):
        area = cv2.contourArea(contour)
        if area < 300 or area > 0.9 * image_area:
            return None  # too small (noise) or too large (background leak)

        x, y, w, h = cv2.boundingRect(contour)
        if h == 0:
            return None
        aspect_ratio = w / float(h)
        if not (0.4 <= aspect_ratio <= 2.5):
            return None

        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        if hull_area <= 0:
            return None
        solidity = area / hull_area
        if solidity < 0.35:
            return None

        perimeter = cv2.arcLength(contour, True)
        circularity = 0.0
        if perimeter > 0:
            circularity = 4 * math.pi * area / (perimeter ** 2)

        # Composite score rewards larger, more solid, more regular blobs
        score = area * (0.5 + 0.3 * solidity + 0.2 * min(circularity, 1.0))
        return score, contour, circularity

    def _select_best_contour(self, mask):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None, None

        image_area = mask.shape[0] * mask.shape[1]
        candidates = []
        for c in contours:
            scored = self._score_contour(c, image_area)
            if scored is not None:
                candidates.append(scored)

        if not candidates:
            return None, None

        best_score, best_contour, best_circularity = max(candidates, key=lambda t: t[0])
        return best_contour, best_circularity

    def _build_final_mask(self, contour, circularity, mask_shape):
        final_mask = np.zeros(mask_shape, dtype=np.uint8)
        epsilon = 0.03 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        # Round/octagonal signs (>=6 vertices, high circularity) are
        # reconstructed as a clean circle from the moment centroid;
        # everything else keeps its (triangle/diamond/rect) polygon.
        if len(approx) >= 6 and circularity > 0.6:
            m = cv2.moments(contour)
            if m["m00"] != 0:
                cx, cy = int(m["m10"] / m["m00"]), int(m["m01"] / m["m00"])
                area = cv2.contourArea(contour)
                radius = int(math.sqrt(area / math.pi) * 1.02)
                cv2.circle(final_mask, (cx, cy), radius, 255, thickness=cv2.FILLED)
                return final_mask
        cv2.drawContours(final_mask, [approx], -1, 255, thickness=cv2.FILLED)
        return final_mask

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def segment(self, bgr_image, color_name):
        """Segment a single BGR image for the given color.

        Returns (rgb_original, cropped_rgb, mask, status) where status is
        one of:
            "ok"       — a reliable sign region was found and cropped
            "fallback" — no reliable region (mask too small / not found);
                         cropped_rgb is the resized ORIGINAL image, and
                         mask is left as an all-white mask of the same
                         size (so downstream code can still treat
                         "cropped" uniformly as "the RGB image to use").

        A fallback is used instead of discarding the image so the sample
        isn't lost from the dataset — it just isn't trimmed down to a
        (likely wrong) tiny region.
        """
        if self.resize_to is not None:
            bgr_image = cv2.resize(bgr_image, self.resize_to)

        rgb_original = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        frame_area = rgb_original.shape[0] * rgb_original.shape[1]

        hsv = self._normalize_illumination(bgr_image)
        raw_mask = self._color_mask(hsv, color_name)
        clean_mask = self._clean_mask(raw_mask)

        best_contour, circularity = self._select_best_contour(clean_mask)

        final_mask = None
        if best_contour is not None:
            final_mask = self._build_final_mask(best_contour, circularity, clean_mask.shape)
            mask_area_ratio = cv2.countNonZero(final_mask) / float(frame_area)
        else:
            mask_area_ratio = 0.0

        if best_contour is None or mask_area_ratio < self.min_mask_area_ratio:
            # No contour passed validation, or the detected blob is too
            # small a fraction of the frame to trust as "the sign" —
            # fall back to the full resized original instead of cropping.
            fallback_mask = np.full(clean_mask.shape, 255, dtype=np.uint8)
            return rgb_original, rgb_original.copy(), fallback_mask, "fallback"

        # Feather the mask edge slightly so the crop doesn't have hard
        # jagged boundaries (helps downstream classifiers).
        feathered = cv2.GaussianBlur(final_mask, (5, 5), 0)
        alpha = (feathered.astype(np.float32) / 255.0)
        alpha_3ch = cv2.merge([alpha, alpha, alpha])
        cropped_rgb = (rgb_original.astype(np.float32) * alpha_3ch).astype(np.uint8)

        return rgb_original, cropped_rgb, final_mask, "ok"


# ----------------------------------------------------------------------
# Dataset-level pipeline
# ----------------------------------------------------------------------
def process_dataset(root_dir, split="Train", output_root="cropped", show=True, resize_to=(300, 300)):
    """Walk <root_dir>/<split>/<Color>/<class_id>/*.jpg (etc.), segment
    every image, save every cropped sign into `output_root` (mirroring
    the split/color/class_id structure) and return the results.

    Parameters
    ----------
    root_dir : str or Path — dataset root containing Train/ and Test/
    split : "Train" or "Test"
    output_root : folder name to save cropped signs into (created if needed)
    show : if True, display original/cropped/mask with matplotlib per image
    resize_to : (w, h) to standardize input images to, or None to keep original size

    Returns
    -------
    list of dicts: {"filename": str, "original": np.ndarray RGB,
                     "cropped": np.ndarray RGB, "status": "ok" | "fallback",
                     "hog": np.ndarray, "color_histogram": np.ndarray,
                     "hog_color": np.ndarray}
        "hog" and "color_histogram" come from feature_extraction_two.py
        and feature_extraction_one.py respectively (run on the cropped
        sign), and "hog_color" is their concatenation.
    """
    root_path = Path(root_dir)
    split_path = root_path / split
    output_path = Path(output_root)

    if not split_path.exists():
        raise FileNotFoundError(f"Split folder not found: {split_path}")

    segmenter = TrafficSignSegmenter(resize_to=resize_to)
    results = []

    for color_name in COLORS:
        color_path = split_path / color_name
        if not color_path.exists():
            print(f"Skipping missing color folder: {color_path}")
            continue

        class_dirs = sorted([d for d in color_path.iterdir() if d.is_dir()])
        for class_dir in class_dirs:
            image_paths = sorted(
                p for p in class_dir.iterdir()
                if p.is_file() and p.suffix.lower() in VALID_EXTENSIONS
            )
            if not image_paths:
                continue

            save_dir = output_path / split / color_name / class_dir.name
            save_dir.mkdir(parents=True, exist_ok=True)

            print(f"Processing {split}/{color_name}/{class_dir.name} "
                  f"({len(image_paths)} images)...")

            for img_path in image_paths:
                bgr = cv2.imread(str(img_path))
                if bgr is None:
                    print(f"  Warning: unreadable file skipped -> {img_path.name}")
                    continue

                rgb_original, cropped_rgb, mask, status = segmenter.segment(bgr, color_name)

                # Both "ok" and "fallback" are saved — a fallback keeps
                # the sample in the dataset (as the full resized image)
                # instead of dropping it entirely.
                save_file = save_dir / img_path.name
                cropped_bgr = cv2.cvtColor(cropped_rgb, cv2.COLOR_RGB2BGR)
                cv2.imwrite(str(save_file), cropped_bgr)

                if status == "fallback":
                    print(f"  Fallback (mask too small/absent) -> {img_path.name}")

                # Feature extraction runs on the saved cropped image, reusing
                # feature_extraction_one.py (color histogram) and
                # feature_extraction_two.py (HOG) rather than duplicating
                # that logic here.
                color_hist = feat1.extract_color_histogram(cropped_bgr)
                hog_vector, _hog_image, _resized = feat2.extract_hog_features(str(save_file))
                if hog_vector is None:
                    hog_vector = np.array([])
                hog_color = np.concatenate([hog_vector, color_hist])

                results.append({
                    "filename": img_path.name,
                    "original": rgb_original,
                    "cropped": cropped_rgb,
                    "status": status,
                    "hog": hog_vector,
                    "color_histogram": color_hist,
                    "hog_color": hog_color,
                })

                if show:
                    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
                    axes[0].imshow(rgb_original)
                    axes[0].set_title("Original")
                    axes[0].axis("off")
                    axes[1].imshow(cropped_rgb)
                    axes[1].set_title("Cropped Sign")
                    axes[1].axis("off")
                    axes[2].imshow(mask, cmap="gray")
                    axes[2].set_title("Mask")
                    axes[2].axis("off")
                    fig.suptitle(img_path.name)
                    plt.tight_layout()
                    plt.show()

    print(f"Done. {len(results)} images processed, saved under '{output_path}/{split}'.")
    return results


if __name__ == "__main__":
    # Example usage — adjust root_dir to your dataset location.
    DATASET_ROOT = "dataset"
    all_results = process_dataset(
        root_dir=DATASET_ROOT,
        split="Train",
        output_root="cropped",
        show=False,
    )
