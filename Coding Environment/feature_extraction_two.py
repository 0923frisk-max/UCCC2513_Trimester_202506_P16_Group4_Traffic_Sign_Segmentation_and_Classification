import cv2
import os
import csv
import time
import numpy as np
from skimage.feature import hog
from skimage import exposure
import matplotlib.pyplot as plt

def extract_hog_features(image_path, visualize=True):
    """Reads an image, preprocesses it, and extracts HOG features.

    Parameters are unchanged from the original implementation:
      64x64 input, 9 orientations, 8x8 pixel cells, 2x2 cell blocks, L2-Hys normalisation.
    """
    img = cv2.imread(image_path)
    if img is None:
        return None, None, None

    img_resized = cv2.resize(img, (64, 64))
    gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)

    if visualize:
        features, hog_image = hog(gray, orientations=9, pixels_per_cell=(8, 8),
                                  cells_per_block=(2, 2), visualize=True,
                                  block_norm='L2-Hys')
    else:
        features = hog(gray, orientations=9, pixels_per_cell=(8, 8),
                       cells_per_block=(2, 2), visualize=False,
                       block_norm='L2-Hys')
        hog_image = None

    return features, hog_image, img_resized

def process_and_visualize(input_dir, output_dir, num_visualizations=30,
                          show_inline=False):
    """Extracts HOG features from every image, measures per-image time,
    and saves a visualisation for the first `num_visualizations` images.

    Returns (successful_extractions, total_images, times_ms, sampled_files).
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # sorted() makes the processing order explicit and repeatable.
    image_files = sorted(f for f in os.listdir(input_dir)
                         if f.lower().endswith(('.jpg', '.png')))

    # Warm-up: the first call to skimage's hog() includes one-off setup cost,
    # which would otherwise be charged to the first image in the timings.
    if image_files:
        extract_hog_features(os.path.join(input_dir, image_files[0]), visualize=False)

    successful_extractions = 0
    total_images = len(image_files)
    times_ms = []
    timing_rows = []
    failed_files = []
    sampled_files = []
    feature_length = None

    for idx, filename in enumerate(image_files):
        img_path = os.path.join(input_dir, filename)

        # --- timed extraction (no visualisation) -------------------------
        start = time.perf_counter()
        features, _, _ = extract_hog_features(img_path, visualize=False)
        elapsed_ms = (time.perf_counter() - start) * 1000.0

        if features is None:
            failed_files.append(filename)
            continue

        successful_extractions += 1
        times_ms.append(elapsed_ms)
        timing_rows.append((filename, elapsed_ms))
        if feature_length is None:
            feature_length = len(features)

        # --- visualisation for the qualitative sample --------------------
        if idx < num_visualizations:
            sampled_files.append(filename)
            _, hog_image, original_resized = extract_hog_features(img_path, visualize=True)

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)

            ax1.axis('off')
            ax1.imshow(cv2.cvtColor(original_resized, cv2.COLOR_BGR2RGB))
            ax1.set_title('Original Image')

            ax2.axis('off')
            hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))
            ax2.imshow(hog_image_rescaled, cmap=plt.cm.gray)
            ax2.set_title(f'HOG Features - Image {idx + 1}')

            plt.savefig(os.path.join(output_dir, f'hog_result_{idx + 1}.png'),
                        dpi=110, bbox_inches='tight')
            if show_inline:
                plt.show()
            plt.close(fig)

    # --- per-image timings written out for the appendix ------------------
    with open(os.path.join(output_dir, 'hog_extraction_times.csv'), 'w', newline='') as fh:
        writer = csv.writer(fh)
        writer.writerow(['image_index', 'filename', 'extraction_time_ms'])
        for i, (fname, t) in enumerate(timing_rows):
            writer.writerow([i + 1, fname, f'{t:.4f}'])

    print(f'Feature vector length: {feature_length}')
    if failed_files:
        print(f'Files that could not be read: {len(failed_files)}')
        for f in failed_files[:10]:
            print('  ', f)

    return successful_extractions, total_images, times_ms, sampled_files

TEST_INPUT_DIRECTORY = "dataset/Test"
VISUALIZATION_OUTPUT_DIRECTORY = "hog_visualizations"
NUM_VISUALIZATIONS = 30

print(f"Processing all images in {TEST_INPUT_DIRECTORY} ...")

run_start = time.perf_counter()
success_count, total, times_ms, sampled_files = process_and_visualize(
    TEST_INPUT_DIRECTORY,
    VISUALIZATION_OUTPUT_DIRECTORY,
    num_visualizations=NUM_VISUALIZATIONS,
)
run_wall_s = time.perf_counter() - run_start

if total > 0:
    processing_rate = (success_count / total) * 100
    t = np.array(times_ms)

    print("\n--- HOG feature extraction: processing outcome ---")
    print(f"Images found                  : {total}")
    print(f"Images successfully processed : {success_count}")
    print(f"Images that failed to load    : {total - success_count}")
    print(f"Processing success rate       : {processing_rate:.2f}%")

    print("\n--- HOG feature extraction: time per image (ms) ---")
    print(f"Mean            : {t.mean():.3f} ms")
    print(f"Median          : {np.median(t):.3f} ms")
    print(f"95th percentile : {np.percentile(t, 95):.3f} ms")
    print(f"Worst case      : {t.max():.3f} ms")
    print(f"Best case       : {t.min():.3f} ms")
    print(f"Total wall-clock for the run : {run_wall_s:.1f} s "
          f"(includes {len(sampled_files)} visualisations)")
    print(f"\nHeadroom against the 2000 ms per-image requirement: "
          f"{2000 / t.mean():.0f}x on the mean, {2000 / t.max():.0f}x on the worst case")


def build_contact_sheet(input_dir, filenames, output_path, cols=5):
    """Renders original / HOG pairs for `filenames` into a single figure."""
    n = len(filenames)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols * 2, figsize=(cols * 4, rows * 2.2))
    axes = np.atleast_2d(axes)

    for ax in axes.ravel():
        ax.axis('off')

    for i, fname in enumerate(filenames):
        r, c = divmod(i, cols)
        _, hog_image, original_resized = extract_hog_features(
            os.path.join(input_dir, fname), visualize=True)

        ax_img = axes[r, c * 2]
        ax_hog = axes[r, c * 2 + 1]

        ax_img.imshow(cv2.cvtColor(original_resized, cv2.COLOR_BGR2RGB))
        ax_img.set_title(f'{i + 1}', fontsize=8)
        ax_hog.imshow(exposure.rescale_intensity(hog_image, in_range=(0, 10)),
                      cmap=plt.cm.gray)

    plt.tight_layout()
    plt.savefig(output_path, dpi=130, bbox_inches='tight')
    plt.show()
    plt.close(fig)
    print(f'Saved contact sheet to {output_path}')


build_contact_sheet(
    TEST_INPUT_DIRECTORY,
    sampled_files,
    os.path.join(VISUALIZATION_OUTPUT_DIRECTORY, 'hog_contact_sheet.png'),
)

scoring_path = os.path.join(VISUALIZATION_OUTPUT_DIRECTORY, 'qualitative_scores.csv')

if not os.path.exists(scoring_path):
    with open(scoring_path, 'w', newline='') as fh:
        writer = csv.writer(fh)
        writer.writerow(['image_number', 'filename', 'verdict', 'reason'])
        for i, fname in enumerate(sampled_files):
            writer.writerow([i + 1, fname, '', ''])
    print(f'Created {scoring_path} — fill in the verdict column, then run the next cell.')
else:
    print(f'{scoring_path} already exists; not overwriting.')


rows = []
with open(scoring_path, newline='') as fh:
    for row in csv.DictReader(fh):
        rows.append(row)

def bucket(v):
    v = v.strip().lower()
    if v.startswith('a'):
        return 'acceptable'
    if v.startswith('b'):
        return 'marginal'
    if v.startswith('d'):
        return 'degraded'
    return None

scored = [(r, bucket(r['verdict'])) for r in rows]
scored = [(r, b) for r, b in scored if b]

counts = {'acceptable': 0, 'marginal': 0, 'degraded': 0}
for _, b in scored:
    counts[b] += 1

n = len(scored)
print(f'Sample size scored : {n} of {len(rows)}')
for k in ('acceptable', 'marginal', 'degraded'):
    print(f'{k.capitalize():<12}: {counts[k]} ({counts[k] / n * 100:.1f}%)' if n else k)

for label in ('marginal', 'degraded'):
    print(f'\n{label.capitalize()} cases:')
    for r, b in scored:
        if b == label:
            print(f"  Image {r['image_number']} ({r['filename']}): {r['reason']}")


def compare_cell_sizes(input_dir, sample_size=300,
                       cell_sizes=((4, 4), (8, 8), (16, 16))):
    image_files = sorted(f for f in os.listdir(input_dir)
                         if f.lower().endswith(('.jpg', '.png')))[:sample_size]

    print(f'{"Cell size":<12}{"Feature dim":>13}{"Mean ms":>10}{"Worst ms":>10}')
    print('-' * 45)

    for ppc in cell_sizes:
        times, dim = [], None
        for fname in image_files:
            img = cv2.imread(os.path.join(input_dir, fname))
            if img is None:
                continue
            gray = cv2.cvtColor(cv2.resize(img, (64, 64)), cv2.COLOR_BGR2GRAY)

            start = time.perf_counter()
            feats = hog(gray, orientations=9, pixels_per_cell=ppc,
                        cells_per_block=(2, 2), visualize=False, block_norm='L2-Hys')
            times.append((time.perf_counter() - start) * 1000.0)
            dim = len(feats)

        t = np.array(times)
        print(f'{str(ppc):<12}{dim:>13}{t.mean():>10.3f}{t.max():>10.3f}')


compare_cell_sizes(TEST_INPUT_DIRECTORY)
