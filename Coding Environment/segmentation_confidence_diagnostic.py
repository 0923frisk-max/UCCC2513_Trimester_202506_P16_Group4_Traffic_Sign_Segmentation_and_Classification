"""
Diagnostic: does segmentation confidence (not class imbalance) explain the
test-set accuracy drop?

Buckets every test prediction into 3 tiers based on the ORIGINAL
segmentation result for that file (from `test_segmented`, the list
returned by `traffic_sign_segmentation.process_dataset`):

  Tier 1 "confident"  : status == "ok"        and mask_area_ratio >  0.30
  Tier 2 "borderline"  : status == "ok"        and 0.05 <= mask_area_ratio <= 0.30
  Tier 3 "fallback"    : status == "fallback"  (full resized original, no crop)

Then reports accuracy / weighted-F1 per tier, for a given trained
model's test_output (e.g. test_output_hog, test_output_color,
test_output_combined).

Usage (paste into the notebook, after test_segmented / test_output_* exist):

    tier_report(test_segmented, test_output_hog, test_paths_valid)
    tier_report(test_segmented, test_output_color, test_paths_valid)
    tier_report(test_segmented, test_output_combined, test_paths_valid)
"""

import os
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score


def _assign_tier(status, mask_area_ratio, borderline_upper=0.30):
    if status == "fallback":
        return "3_fallback"
    if mask_area_ratio > borderline_upper:
        return "1_confident"
    return "2_borderline"


def build_filename_lookup(segmented_results):
    """filename -> {"status": ..., "mask_area_ratio": ...} from a
    process_dataset() results list (e.g. test_segmented)."""
    lookup = {}
    for item in segmented_results:
        lookup[item["filename"]] = {
            "status": item["status"],
            "mask_area_ratio": item["mask_area_ratio"],
        }
    return lookup


def tier_report(segmented_results, test_output, test_image_paths, borderline_upper=0.30):
    """Print + return a per-tier accuracy/F1 breakdown for one trained
    model's test_output.

    Parameters
    ----------
    segmented_results : the list returned by process_dataset for the SAME
        split (e.g. test_segmented) -- used to look up each file's
        status/mask_area_ratio.
    test_output : dict returned by test_on_feature(...), e.g.
        test_output_hog / test_output_color / test_output_combined.
    test_image_paths : the path list aligned with test_output["y_test"] /
        test_output["test_preds"] (this is `test_paths_valid` in the
        notebook).
    borderline_upper : mask_area_ratio above which an "ok" result counts
        as "confident" rather than "borderline".
    """
    lookup = build_filename_lookup(segmented_results)

    y_true = np.asarray(test_output["y_test"])
    y_pred = np.asarray(test_output["test_preds"])

    tiers = []
    missing = 0
    for path in test_image_paths:
        fname = os.path.basename(path)
        meta = lookup.get(fname)
        if meta is None:
            missing += 1
            tiers.append("unknown")
            continue
        tiers.append(_assign_tier(meta["status"], meta["mask_area_ratio"], borderline_upper))
    tiers = np.array(tiers)

    if missing:
        print(f"[warn] {missing} test file(s) had no matching entry in "
              f"segmented_results (filename mismatch?) -- excluded below.")

    rows = []
    for tier_name in ["1_confident", "2_borderline", "3_fallback"]:
        mask = tiers == tier_name
        n = int(mask.sum())
        if n == 0:
            rows.append({"tier": tier_name, "n_images": 0, "accuracy": None, "f1_weighted": None})
            continue
        acc = accuracy_score(y_true[mask], y_pred[mask])
        f1 = f1_score(y_true[mask], y_pred[mask], average="weighted", zero_division=0)
        rows.append({"tier": tier_name, "n_images": n, "accuracy": acc, "f1_weighted": f1})

    report_df = pd.DataFrame(rows)
    print(f"\n=== Segmentation-confidence tiers — {test_output['feature_name']} ===")
    print(report_df.to_string(index=False))
    return report_df
