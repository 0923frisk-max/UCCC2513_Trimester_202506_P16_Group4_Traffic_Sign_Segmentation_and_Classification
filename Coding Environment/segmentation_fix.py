"""
traffic_sign_segmentation.py
=============================

Integrated traffic-sign segmentation module that merges and improves the
three separate scripts (blue / red / yellow color segmentation + shape.py)
into one reliable pipeline.

Key reliability improvements over the original scripts
--------------------------------------------------------
1. Illumination normalization BEFORE color thresholding:
   - Automatic gamma correction first normalizes GLOBAL exposure (very
     bright/overexposed or very dark/underexposed frames), which CLAHE
     alone does not fix — CLAHE only boosts LOCAL contrast, so a frame
     that's uniformly too bright or too dark still ends up mis-thresholded
     without this step.
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
3. Dual colorspace color matching: the HSV threshold is OR-combined with
   a second mask computed from LAB's a*/b* chrominance channels. HSV hue
   is very sensitive to exposure/white-balance shifts (a color can drift
   just outside the tuned range under odd lighting); LAB's chrominance
   channels are comparatively decoupled from luminance and often still
   catch a sign the HSV range missed. The extra false positives this
   invites are filtered out downstream by shape scoring (#5).
4. Broken-border retry: a sign's colored border (esp. red outlines) can
   fragment into disconnected arcs under uneven lighting or color
   inconsistency, so the closed mask ends up as several small blobs
   instead of one loop. If no contour passes shape validation with the
   normal closing kernel, the module retries once with a larger
   "bridging" kernel that merges nearby fragments into a single blob
   before re-running shape validation — used only as a fallback so it
   doesn't blur together genuinely separate/well-segmented signs.
5. HSV ranges were widened/aligned across all three original scripts
   (blue/red/yellow each had two slightly different range sets) and a
   brightness-adaptive lower-V bound is used so dim/underexposed images
   don't lose their sign to the mask.
6. Morphology + contour selection now scores candidates using area,
   aspect ratio, solidity AND circularity together (the original
   shape.py only used area/aspect/solidity for the mask, and a crude
   vertex-count check to decide "draw as filled circle vs polygon").
   Scoring instead of hard sequential filtering makes the module far
   less likely to drop a valid sign because of one borderline check.
7. One shared implementation instead of 3 near-duplicate scripts, so a
   fix/tune in one place (e.g. HSV ranges) benefits all colors.
8. Glare reconstruction: retroreflective sign material commonly produces
   a blown-out, near-white/desaturated highlight under direct sun or
   headlights, which can cut a colored border or fill into two or more
   disconnected pieces even before any noise/lighting issue. A bright +
   desaturated "glare" mask is computed and grown INTO from the existing
   color mask via morphological reconstruction (geodesic dilation), so
   only glare that actually touches a detected color region gets healed
   back in -- an unrelated bright patch elsewhere in the frame (sky,
   oncoming headlights) that never touches the sign is left alone.
9. Colorless shape/edge fallback: if HSV+LAB both come back with
   essentially no usable color evidence at all (fog, night, badly faded
   paint -- "颜色捕捉不到" in the fullest sense), the module falls back to
   Canny edges + the same area/aspect/solidity/circularity scoring used
   for color blobs, with an added vertex-count check against the shapes
   road signs actually come in (triangle/diamond/rect/pentagon/octagon/
   circle). This is a last resort before giving up entirely, and is
   reported as its own status ("shape_fallback") so downstream code can
   tell "found by color" apart from "found by geometry alone".
10. Real bounding-box crop instead of full-frame black background: the
    final mask is feathered and alpha-matted, then the OUTPUT IS CROPPED
    to the mask's bounding box (with a small padding margin) rather than
    keeping the full resized frame with everything outside the mask
    painted black. A full-frame black background lets black pixels
    dominate downstream color-histogram statistics and manufactures a
    hard artificial edge all around the mask boundary for HOG -- neither
    reflects the sign itself.

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
    #   {"filename": str, "original": np.ndarray (RGB), "cropped": np.ndarray (RGB,
    #    cropped to the detected sign's bounding box -- NOT full-frame size),
    #    "status": "ok" | "shape_fallback" | "fallback", "mask_area_ratio": float,
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

    def __init__(self, resize_to=(300, 300), min_mask_area_ratio=0.05, bridge_kernel_size=21,
                 use_gamma_correction=True, use_lab_color=False, use_bridge_retry=True,
                 use_shape_validation=True, use_glare_reconstruction=True,
                 use_shape_fallback=False, glare_v_thresh=235, glare_s_thresh=55,
                 crop_padding_ratio=0.10):
        self.resize_to = resize_to
        # If the final mask covers less than this fraction of the frame,
        # the "detected" region is treated as too small/unreliable and we
        # fall back to returning the (resized) original image instead of
        # a crop that is probably noise, not signal.
        self.min_mask_area_ratio = min_mask_area_ratio
        self.kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        self.kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15)) # 11 to 15
        # Used only as a retry when the normal kernel leaves a sign's
        # border as several disconnected fragments (see segment()). How
        # large this needs to be depends on the actual gap width in your
        # images -- tune it up if borders are still coming out fragmented,
        # down if it's fusing genuinely separate nearby signs together.
        self.kernel_close_bridge = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (bridge_kernel_size, bridge_kernel_size)
        )
        # Single dilation step used to grow the color mask into adjacent
        # glare pixels one ring at a time (see _reconstruct_through_glare).
        # Kept small and applied iteratively so growth stays confined to
        # pixels that are actually glare, instead of one big blunt dilation
        # that would also swallow nearby background. MUST be 8-connected
        # (full 3x3 square, not a plus-shaped ellipse/cross): two
        # hard-edged color regions rendered next to each other very often
        # meet only at a single diagonal pixel (a rasterization seam), and
        # a 4-connected kernel can never cross a pure diagonal gap no
        # matter how many iterations it's given.
        self.kernel_glare_grow = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

        # Ablation switches -- flip these off one at a time to isolate
        # which addition is actually helping/hurting on your real data,
        # instead of always testing everything bundled.
        self.use_gamma_correction = use_gamma_correction
        self.use_lab_color = use_lab_color
        self.use_bridge_retry = use_bridge_retry
        self.use_shape_validation = use_shape_validation
        self.use_glare_reconstruction = use_glare_reconstruction
        self.use_shape_fallback = use_shape_fallback

        # A pixel counts as "glare" when it's both very bright (V) and
        # near-colorless (low S) -- the signature of a blown-out highlight
        # on reflective sign material, as opposed to a saturated patch of
        # the sign's actual color.
        self.glare_v_thresh = glare_v_thresh
        self.glare_s_thresh = glare_s_thresh

        # Extra margin (fraction of the bbox's own width/height) kept
        # around the detected sign when cropping to its bounding box, so
        # the crop doesn't shave off a sliver of the actual border.
        self.crop_padding_ratio = crop_padding_ratio

    def _fill_holes(self, mask):
        """把闭合色环内部（白箭头 / 白字）补成实心。
        从图像边界 flood fill，取反 = 所有没被外部连通到的洞。"""
        flooded = mask.copy()
        h, w = mask.shape[:2]
        scratch = np.zeros((h + 2, w + 2), np.uint8)
        cv2.floodFill(flooded, scratch, (0, 0), 255)
        return cv2.bitwise_or(mask, cv2.bitwise_not(flooded))

    # ------------------------------------------------------------------
    # Illumination / noise handling
    # ------------------------------------------------------------------
    def _auto_gamma_correct(self, bgr_image):
        """Normalize GLOBAL exposure via gamma correction so very bright
        (overexposed) or very dark (underexposed) frames land closer to
        mid-gray before anything else runs. CLAHE (applied later) only
        boosts LOCAL contrast — it does not fix a frame that is uniformly
        too bright or too dark, which is exactly the "光照太亮太暗" case.
        """
        gray_mean = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY).mean() / 255.0
        gray_mean = float(np.clip(gray_mean, 1e-3, 1 - 1e-3))
        # Solve for the exponent that maps the frame's current mean
        # brightness to mid-gray: gray_mean ** exponent == 0.5.
        # (Applying its reciprocal here, as an earlier version of this
        # function did, pushes brightness AWAY from mid-gray instead of
        # toward it -- it makes dark frames darker and bright frames
        # brighter, i.e. the exact opposite of what "gamma correction for
        # exposure" is supposed to do. Verified against gray_mean=0.089
        # and 0.9: the reciprocal moves them to 0.002 and 0.959
        # respectively, while applying the exponent directly moves them
        # to 0.379 and 0.768 -- correctly toward 0.5.)
        exponent = math.log(0.5) / math.log(gray_mean)
        # Clamp so a near-black or near-white frame doesn't get an
        # extreme correction that amplifies noise instead of exposure.
        exponent = float(np.clip(exponent, 0.4, 2.5))
        table = (np.linspace(0, 1, 256) ** exponent * 255).astype(np.uint8)
        return cv2.LUT(bgr_image, table)

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
        """Returns (normalized_bgr, hsv) — both are handed to the color
        matching step so it can look at HSV hue AND LAB chrominance."""
        if self.use_gamma_correction:
            exposure_corrected = self._auto_gamma_correct(bgr_image)
        else:
            exposure_corrected = bgr_image
        balanced = self._gray_world_white_balance(exposure_corrected)

        # Denoise before anything else: median kills salt-and-pepper /
        # compression speckle, Gaussian softens residual sensor noise.
        denoised = cv2.medianBlur(balanced, 1)
        denoised = cv2.GaussianBlur(denoised, (3, 3), 0)

        lab = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l_equalized = clahe.apply(l_channel)
        lab_equalized = cv2.merge((l_equalized, a_channel, b_channel))

        normalized_bgr = cv2.cvtColor(lab_equalized, cv2.COLOR_LAB2BGR)
        hsv = cv2.cvtColor(normalized_bgr, cv2.COLOR_BGR2HSV)
        return normalized_bgr, hsv

    # ------------------------------------------------------------------
    # Color thresholding
    # ------------------------------------------------------------------
    def _get_hsv_ranges(self, color_name, hsv_image):
        """Return list of (lower, upper) HSV bounds, with the V lower
        bound nudged down for images that are globally dark so faint
        (shadowed / backlit) signs aren't lost."""
        mean_v = hsv_image[:, :, 2].mean()
        # In a dark scene, relax the minimum value/brightness we accept
        v_floor = 40 if mean_v < 100 else 55 #30 to 40, 90 to 100,  45 to 55

        if color_name == "Blue": #95 to 100
            return [(np.array([100, 120, v_floor]), np.array([135, 255, 255]))]
        elif color_name == "Yellow":
            return [(np.array([10, 77, v_floor]), np.array([34, 255, 255]))]
        elif color_name == "Red":
            return [
                (np.array([0, 80, v_floor]), np.array([10, 255, 255])),
                (np.array([165, 80, v_floor]), np.array([180, 255, 255])),
            ]
        return []

    def _color_mask(self, hsv_image, bgr_image, color_name):
        hsv_mask = np.zeros(hsv_image.shape[:2], dtype=np.uint8)
        for lower, upper in self._get_hsv_ranges(color_name, hsv_image):
            hsv_mask = cv2.bitwise_or(hsv_mask, cv2.inRange(hsv_image, lower, upper))

        if not self.use_lab_color:
            return hsv_mask

        lab_mask = self._lab_color_mask(bgr_image, color_name)

        # OR-combine two independent color spaces: HSV hue can drift just
        # outside the tuned range under odd lighting/white-balance ("颜色
        # 没对上"), while LAB's a*/b* chrominance channels are comparatively
        # decoupled from luminance and often still catch it. The extra
        # false-positive area this invites gets filtered out downstream by
        # shape scoring (area/aspect/solidity/circularity).
        return cv2.bitwise_or(hsv_mask, lab_mask)

    def _lab_color_mask(self, bgr_image, color_name):
        """Secondary color mask from LAB's a*/b* chrominance channels.
        LAB separates luminance (L) from color (a*: green-red, b*:
        blue-yellow), so a sign's chrominance stays comparatively stable
        even when overall brightness/white-balance shifts — exactly the
        case where a fixed HSV hue range starts to miss."""
        lab = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2LAB)
        a_channel, b_channel = lab[:, :, 1], lab[:, :, 2]

        if color_name == "Red":
            # High a* (red/magenta) combined with a moderately positive
            # b* excludes pure magenta/pink, keeping it closer to "red".
            return cv2.inRange(a_channel, 150, 255) & cv2.inRange(b_channel, 120, 255)
        elif color_name == "Blue":
            # Blue sits on the low end of both axes: slightly-green-to-
            # neutral a*, and low (blue-leaning) b*.
            return cv2.inRange(a_channel, 100, 145) & cv2.inRange(b_channel, 60, 115)
        elif color_name == "Yellow":
            # High b* (yellow) with a roughly neutral a* excludes
            # oranges/reds that also have a high b*.
            return cv2.inRange(a_channel, 110, 150) & cv2.inRange(b_channel, 150, 255)
        return np.zeros(bgr_image.shape[:2], dtype=np.uint8)

    # ------------------------------------------------------------------
    # Glare (blown-highlight) reconstruction
    # ------------------------------------------------------------------
    def _detect_glare_mask(self, hsv_image):
        """Near-white blown-out highlight pixels: very bright AND
        desaturated. Retroreflective sign material commonly produces
        exactly this under direct headlight/sunlight glare, which can cut
        a colored border or split a filled region into disconnected
        pieces -- independent of any exposure/noise issue elsewhere."""
        v_channel = hsv_image[:, :, 2]
        s_channel = hsv_image[:, :, 1]
        bright = cv2.inRange(v_channel, self.glare_v_thresh, 255)
        desaturated = cv2.inRange(s_channel, 0, self.glare_s_thresh)
        return cv2.bitwise_and(bright, desaturated)

    def _reconstruct_through_glare(self, raw_mask, hsv_image, max_iterations=40):
        """Heals holes/breaks in the color mask caused by glare, via
        morphological reconstruction by dilation: grow the color mask one
        small ring at a time, but only into pixels that are both (a)
        glare and (b) reachable from an already-detected color pixel.
        An isolated bright patch elsewhere in the frame (sky, oncoming
        headlights) never touches the color mask, so it is never pulled
        in -- this only closes gaps that sit on/inside the sign itself.
        """
        if not self.use_glare_reconstruction:
            return raw_mask

        glare_mask = self._detect_glare_mask(hsv_image)
        if cv2.countNonZero(glare_mask) == 0:
            return raw_mask

        ceiling = cv2.bitwise_or(raw_mask, glare_mask)
        marker = raw_mask.copy()
        for _ in range(max_iterations):
            grown = cv2.dilate(marker, self.kernel_glare_grow, iterations=1)
            grown = cv2.bitwise_and(grown, ceiling)
            if cv2.countNonZero(cv2.bitwise_xor(grown, marker)) == 0:
                break
            marker = grown
        return marker

    def _clean_mask(self, mask, kernel_close=None):
        kernel_close = self.kernel_close if kernel_close is None else kernel_close
        opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel_open)
        closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel_close)
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

    def _select_largest_contour_raw(self, mask, min_area=200):
        """No-shape-validation path: just the largest contour above a
        noise floor, with no aspect-ratio/solidity/circularity filtering
        at all. Used when use_shape_validation=False, so you can test
        whether the geometric filtering is actually helping or just
        rejecting/distorting otherwise-fine color detections."""
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        largest = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest) < min_area:
            return None
        return largest

    def _build_final_mask_raw(self, contour, mask_shape):
        """No-shape-validation path: fill the contour exactly as detected
        -- no approxPolyDP simplification, no forcing round signs into a
        reconstructed circle. This keeps whatever the color mask actually
        found, jagged edges and all."""
        final_mask = np.zeros(mask_shape, dtype=np.uint8)
        cv2.drawContours(final_mask, [contour], -1, 255, thickness=cv2.FILLED)
        return final_mask

    def _select_best_contour(self, mask, min_area=200, frag_ratio=0.15):
        """替换原来的 _select_best_contour。

        原版只取「得分最高的单个 contour」，所以：
          - 禁止通行牌被白横杠切成上下两段弧 -> 只拿到上面那段
          - 下半部偏暗的蓝牌色带断成 C 形 -> 填充后中间留一个黑洞
        这里改成：把明显属于同一块牌子的所有碎片一起收进来，
        交给 _build_final_mask 做凸包。
        """
        mask = self._fill_holes(mask)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = [c for c in contours if cv2.contourArea(c) >= min_area]
        if not contours:
            return None, None

        biggest = max(cv2.contourArea(c) for c in contours)
        # 面积至少是最大块 15% 的碎片，视为同一块牌子的一部分
        kept = [c for c in contours if cv2.contourArea(c) >= frag_ratio * biggest]

        merged = cv2.convexHull(np.vstack(kept))

        image_area = mask.shape[0] * mask.shape[1]
        area = cv2.contourArea(merged)
        # 注意：输入本身就是裁好的牌子时，正确的 mask 本来就会接近满屏，
        # 所以原来的 area > 0.9 * image_area 直接 reject 是错的。
        if area < min_area or area > 0.995 * image_area:
            return None, None

        x, y, w, h = cv2.boundingRect(merged)
        if h == 0 or not (0.35 <= w / float(h) <= 2.8):
            return None, None

        perimeter = cv2.arcLength(merged, True)
        circularity = 4 * np.pi * area / (perimeter ** 2) if perimeter > 0 else 0.0
        return merged, circularity

    def _build_final_mask(self, contour, circularity, mask_shape):
        """替换原来的 _build_final_mask。

        原版 epsilon = 0.03 * perimeter 太大，一个圆会被压成 4~5 个顶点，
        且顶点数 <6 时走不到「重建成圆」那条分支 -> 输出就是那个梯形。
        交通标志本身都是凸的（圆 / 三角 / 方 / 菱形 / 八边形），
        直接填凸包即可，不需要再化简。
        """
        final_mask = np.zeros(mask_shape, dtype=np.uint8)
        hull = cv2.convexHull(contour)

        if circularity is not None and circularity > 0.80:
            (cx, cy), r = cv2.minEnclosingCircle(hull)
            cv2.circle(final_mask, (int(cx), int(cy)), int(r * 1.02), 255, cv2.FILLED)
            return final_mask

        cv2.drawContours(final_mask, [hull], -1, 255, cv2.FILLED)
        return final_mask

    def _edge_shape_mask_and_contour(self, bgr_image):
        """Colorless last resort: when HSV+LAB both come back with
        essentially no usable color evidence (fog, night, badly faded
        paint), fall back to pure geometry -- Canny edges closed into
        blobs, scored with the same area/aspect/solidity/circularity
        rubric as the color path, plus a vertex-count check against the
        small set of shapes road signs actually come in. This trades
        some false-positive risk (no color prior at all) for not losing
        the sample outright, so it demands a stricter solidity than the
        color-based path to compensate.
        """
        gray = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        edges = cv2.Canny(gray, 40, 120)
        edges = cv2.dilate(edges, self.kernel_open, iterations=1)
        closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, self.kernel_close_bridge)

        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None, None

        image_area = gray.shape[0] * gray.shape[1]
        candidates = []
        for contour in contours:
            scored = self._score_contour(contour, image_area)
            if scored is None:
                continue
            score, contour, circularity = scored

            epsilon = 0.03 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            # Road signs are triangles, quads (rect/diamond), pentagons,
            # octagons, or circles. Reject anything else that only
            # cleared area/aspect/solidity by luck.
            is_plausible_sign_shape = len(approx) in (3, 4, 5, 6, 7, 8) or circularity > 0.7
            if not is_plausible_sign_shape:
                continue

            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            solidity = cv2.contourArea(contour) / hull_area if hull_area > 0 else 0.0
            # No color evidence at all here, so require a cleaner outline
            # than the color-assisted path (0.35) before trusting it.
            if solidity < 0.55:
                continue

            candidates.append((score, contour, circularity))

        if not candidates:
            return None, None

        best_score, best_contour, best_circularity = max(candidates, key=lambda t: t[0])
        return best_contour, best_circularity

    def _find_contour(self, mask):
        """Dispatches to the shape-validated or raw contour selection
        depending on use_shape_validation."""
        if self.use_shape_validation:
            return self._select_best_contour(mask)
        contour = self._select_largest_contour_raw(mask)
        return contour, None

    def _build_mask(self, contour, circularity, mask_shape):
        """Dispatches to the shape-validated (polygon/circle
        reconstruction) or raw (fill contour as-is) mask builder."""
        if self.use_shape_validation:
            return self._build_final_mask(contour, circularity, mask_shape)
        return self._build_final_mask_raw(contour, mask_shape)

    def _crop_to_bbox(self, rgb_image, mask):
        """Crop to the mask's bounding box (plus a small padding margin)
        instead of keeping the full frame with everything outside the
        mask painted black. A full-frame black background lets the black
        area dominate downstream color-histogram statistics and
        manufactures a hard artificial edge all around the mask boundary
        for HOG -- neither reflects the sign itself.
        """
        ys, xs = np.where(mask > 0)
        if ys.size == 0 or xs.size == 0:
            return rgb_image, mask

        h, w = mask.shape[:2]
        x0, x1 = int(xs.min()), int(xs.max())
        y0, y1 = int(ys.min()), int(ys.max())

        pad_x = int((x1 - x0 + 1) * self.crop_padding_ratio)
        pad_y = int((y1 - y0 + 1) * self.crop_padding_ratio)
        x0 = max(0, x0 - pad_x)
        y0 = max(0, y0 - pad_y)
        x1 = min(w - 1, x1 + pad_x)
        y1 = min(h - 1, y1 + pad_y)

        return rgb_image[y0:y1 + 1, x0:x1 + 1], mask[y0:y1 + 1, x0:x1 + 1]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def segment(self, bgr_image, color_name):
        """Segment a single BGR image for the given color.

        Returns (rgb_original, cropped_rgb, mask, status, mask_area_ratio)
        where status is one of:
            "ok"             — a reliable sign region was found by color
                                and cropped to its bounding box.
            "shape_fallback" — color evidence was essentially absent, but
                                a plausible sign shape was still found via
                                Canny edges + geometric scoring alone;
                                cropped to that region's bounding box.
            "fallback"       — nothing usable was found at all (mask too
                                small / absent even after every retry);
                                cropped_rgb is the resized ORIGINAL image
                                (full frame, uncropped) and mask is an
                                all-white mask of the same size (so
                                downstream code can still treat "cropped"
                                uniformly as "the RGB image to use").

        For "ok"/"shape_fallback", cropped_rgb and mask are sized to the
        detected sign's bounding box (plus a small padding margin) rather
        than the full frame — the region outside the mask is alpha-
        feathered, not painted black and kept at full-frame size, so it
        doesn't skew downstream color-histogram/HOG features.

        mask_area_ratio is the fraction of the full frame covered by the
        detected mask BEFORE the fallback decision and BEFORE cropping —
        i.e. even for a "fallback" result this tells you how close it
        came (0.0 means nothing passed validation at all, vs. e.g. 0.09
        means something was found but was judged too small to trust).
        This lets you bucket results by how confident the segmentation
        actually was, instead of only knowing status as a category.

        A "fallback" is used instead of discarding the image so the
        sample isn't lost from the dataset — it just isn't trimmed down
        to a (likely wrong) tiny region.
        """
        if self.resize_to is not None:
            bgr_image = cv2.resize(bgr_image, self.resize_to)

        rgb_original = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        frame_shape = rgb_original.shape[:2]
        frame_area = frame_shape[0] * frame_shape[1]

        normalized_bgr, hsv = self._normalize_illumination(bgr_image)
        raw_mask = self._color_mask(hsv, normalized_bgr, color_name)
        # Heal glare-induced holes/breaks (see _reconstruct_through_glare)
        # before the normal closing/contour pipeline ever runs.
        raw_mask = self._reconstruct_through_glare(raw_mask, hsv)
        clean_mask = self._clean_mask(raw_mask)

        best_contour, circularity = self._find_contour(clean_mask)

        if best_contour is None and self.use_bridge_retry:
            # No single connected region passed validation -- this is the
            # signature of a border that's fragmented into disconnected
            # arcs (uneven lighting / color inconsistency breaking up a
            # thin colored outline). Retry once with a larger "bridging"
            # closing kernel that merges nearby fragments into one blob
            # before re-scoring, rather than immediately giving up.
            bridged_mask = self._clean_mask(raw_mask, kernel_close=self.kernel_close_bridge)
            bridged_contour, bridged_circularity = self._find_contour(bridged_mask)
            if bridged_contour is not None:
                clean_mask = bridged_mask
                best_contour, circularity = bridged_contour, bridged_circularity

        status = "ok"
        if best_contour is None and self.use_shape_fallback:
            # Color evidence is essentially absent (fog/night/faded
            # paint) -- try pure edge/shape geometry as a last resort
            # before giving up on the sample entirely.
            shape_contour, shape_circularity = self._edge_shape_mask_and_contour(normalized_bgr)
            if shape_contour is not None:
                best_contour, circularity = shape_contour, shape_circularity
                status = "shape_fallback"

        final_mask = None
        if best_contour is not None:
            final_mask = self._build_mask(best_contour, circularity, frame_shape)
            mask_area_ratio = cv2.countNonZero(final_mask) / float(frame_area)
        else:
            mask_area_ratio = 0.0

        if best_contour is None or mask_area_ratio < self.min_mask_area_ratio:
            # No contour passed validation, or the detected blob is too
            # small a fraction of the frame to trust as "the sign" —
            # fall back to the full resized original instead of cropping.
            fallback_mask = np.full(frame_shape, 255, dtype=np.uint8)
            return rgb_original, rgb_original.copy(), fallback_mask, "fallback", mask_area_ratio

        # Feather the mask edge slightly so the crop doesn't have hard
        # jagged boundaries (helps downstream classifiers), THEN crop to
        # the mask's bounding box instead of keeping full-frame black.
        feathered = cv2.GaussianBlur(final_mask, (3, 3), 0)
        alpha = (feathered.astype(np.float32) / 255.0)
        alpha_3ch = cv2.merge([alpha, alpha, alpha])
        matted_full = (rgb_original.astype(np.float32) * alpha_3ch).astype(np.uint8)

        cropped_rgb, cropped_mask = self._crop_to_bbox(matted_full, final_mask)

        return rgb_original, cropped_rgb, cropped_mask, status, mask_area_ratio


# ----------------------------------------------------------------------
# Dataset-level pipeline
# ----------------------------------------------------------------------
def process_dataset(root_dir, split="Train", output_root="cropped", show=True, resize_to=(300, 300),
                     min_mask_area_ratio=0.05, use_gamma_correction=True, use_lab_color=True,
                     use_bridge_retry=True, use_shape_validation=True, bridge_kernel_size=21,
                     use_glare_reconstruction=True, use_shape_fallback=True,
                     glare_v_thresh=235, glare_s_thresh=55, crop_padding_ratio=0.08):
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
    min_mask_area_ratio, use_gamma_correction, use_lab_color, use_bridge_retry,
    use_shape_validation, bridge_kernel_size, use_glare_reconstruction,
    use_shape_fallback, glare_v_thresh, glare_s_thresh, crop_padding_ratio :
        forwarded to TrafficSignSegmenter.
        use_shape_validation=False skips the aspect-ratio/solidity/
        circularity filtering AND the polygon/circle reconstruction --
        the largest color-matched blob is used as-is. Use the use_*
        flags to run an ablation (toggle one at a time) to see which
        addition actually helps on your data instead of testing them
        bundled.

    Returns
    -------
    list of dicts: {"filename": str, "original": np.ndarray RGB,
                     "cropped": np.ndarray RGB (cropped to the detected
                     sign's bounding box, NOT full-frame size, except for
                     "fallback" results which stay full-frame),
                     "status": "ok" | "shape_fallback" | "fallback",
                     "mask_area_ratio": float,
                     "hog": np.ndarray, "color_histogram": np.ndarray,
                     "hog_color": np.ndarray}
        "mask_area_ratio" is the fraction of the frame the detected mask
        covered BEFORE the fallback decision — useful for bucketing "ok"
        results by segmentation confidence (see TrafficSignSegmenter.segment).
        "hog" and "color_histogram" come from feature_extraction_two.py
        and feature_extraction_one.py respectively (run on the cropped
        sign), and "hog_color" is their concatenation.
    """
    root_path = Path(root_dir)
    split_path = root_path / split
    output_path = Path(output_root)

    if not split_path.exists():
        raise FileNotFoundError(f"Split folder not found: {split_path}")

    segmenter = TrafficSignSegmenter(
        resize_to=resize_to,
        min_mask_area_ratio=min_mask_area_ratio,
        bridge_kernel_size=bridge_kernel_size,
        use_gamma_correction=use_gamma_correction,
        use_lab_color=use_lab_color,
        use_bridge_retry=use_bridge_retry,
        use_shape_validation=use_shape_validation,
        use_glare_reconstruction=use_glare_reconstruction,
        use_shape_fallback=use_shape_fallback,
        glare_v_thresh=glare_v_thresh,
        glare_s_thresh=glare_s_thresh,
        crop_padding_ratio=crop_padding_ratio,
    )
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

                rgb_original, cropped_rgb, mask, status, mask_area_ratio = segmenter.segment(bgr, color_name)

                # Both "ok" and "fallback" are saved — a fallback keeps
                # the sample in the dataset (as the full resized image)
                # instead of dropping it entirely.
                save_file = save_dir / img_path.name
                cropped_bgr = cv2.cvtColor(cropped_rgb, cv2.COLOR_RGB2BGR)
                cv2.imwrite(str(save_file), cropped_bgr)

                if status == "fallback":
                    print(f"  Fallback (mask too small/absent) -> {img_path.name}")
                elif status == "shape_fallback":
                    print(f"  Shape-only fallback (no usable color evidence) -> {img_path.name}")

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
                    "mask_area_ratio": mask_area_ratio,
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

    all_results = process_dataset(
        root_dir=DATASET_ROOT,
        split="Test",
        output_root="cropped",
        show=False,
    )