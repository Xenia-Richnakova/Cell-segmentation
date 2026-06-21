import cv2
import numpy as np
from skimage.measure import label, regionprops
#import matplotlib.pyplot as plt

from counting_cells import EdgeFinder
from skimage.draw import line as skline
from skimage.filters import gaussian

#Povodne YPGal_Snap-8123 trval 12.19 s
#Povodne YPGal_Snap-8097 trval 11.10 s
#Povodne SD_Day1_Snap-6908 trval 10.69 s
#Povodne 6 obrazkov v batch spravovani trvalo 52 s -> 8.6 s na obrazok

####### Zmena split_touching_cells_by_defects
#YPGal_Snap-8123 trval 14.00 s
#YPGal_Snap-8097 trval 11.95 s
#SD_Day1_Snap-6908 trval 8.31 s

# get_and_split_all_labels, Cache the gradient, replace of gradient_magnitude, smoothed_gradient added a split_touching_cells_by_defects
#YPGal_Snap-8123 trval 11.00 s
#YPGal_Snap-8097 trval 8.52 s
#SD_Day1_Snap-6908 trval 7.31 s

# get_and_split_all_labels, Cache the gradient, replace of gradient_magnitude, smoothed_gradient added a split_touching_cells_by_defects
# select_the_most_regular, adjust_largest_object_regularity, gradually_select_best

#YPGal_Snap-8123 trval 8.33 s
#YPGal_Snap-8097 trval 7.31 s
#SD_Day1_Snap-6908 trval 5.73 s
# 6 obrazkov v batch spravovani trvalo 34 s -> 5.6 s na obrazok

# ================== Helpers =======================

def get_line_pixels(pt1, pt2):
    x1, y1 = pt1
    x2, y2 = pt2
    rr, cc = skline(y1, x1, y2, x2)   # rows, cols
    return rr, cc


def sample_gradient_on_line(gradient_img, pt1, pt2):
    rr, cc = get_line_pixels(pt1, pt2)
    values = gradient_img[rr, cc]
    coords = list(zip(cc, rr))  # back to (x, y) for plotting if needed
    return coords, values

def split_touching_cells_by_defects(binary_mask, depth_threshold=50.0, gradient_img=None, offset_xy=(0, 0)):
    binary_mask = binary_mask.astype(bool)
    mask_uint8 = (binary_mask.astype(np.uint8) * 255)

    x_offset, y_offset = offset_xy

    contours, _ = cv2.findContours(
        mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
    )
    if not contours:
        return label(binary_mask), [], None, None

    cnt = max(contours, key=cv2.contourArea)

    # hull indices for convexity defects
    hull_idx = cv2.convexHull(cnt, returnPoints=False)
    if hull_idx is None or len(hull_idx) < 3:
        return label(binary_mask), [], None, None

    # hull points for plotting - return for every valid object
    hull_points = cv2.convexHull(cnt, returnPoints=True)
    hull_points = hull_points[:, 0, :]   # shape (N, 2)

    # Translate hull coordinates from ROI coordinates to original-image coordinates.
    hull_points_global = hull_points + np.array([x_offset, y_offset])

    defects = cv2.convexityDefects(cnt, hull_idx)
    if defects is None:
        return label(binary_mask), [], None, hull_points_global

    deep_defects_local = []

    for i in range(defects.shape[0]):
        # start, end point, farthest point, approximate distance to farthest point of the defect
        s, e, f, d = defects[i, 0]
        actual_depth = d / 256.0
        far_point_local = tuple(cnt[f][0])

        if actual_depth > depth_threshold:
            deep_defects_local.append((i, actual_depth, far_point_local))

    # Return global coordinates for visualisation.
    deep_defects_global = [
        (index, depth_value, (point[0] + x_offset, point[1] + y_offset))
        for index, depth_value, point in deep_defects_local
    ]

    if len(deep_defects_local) < 2:
        return label(binary_mask), deep_defects_global, None, hull_points_global

    # Use the two deepest defects.
    deep_defects_local.sort(reverse=True, key=lambda item: item[1])

    idx1, depth1, pt1_local = deep_defects_local[0]
    idx2, depth2, pt2_local = deep_defects_local[1]

    line_gradient = None

    if gradient_img is not None:
        coords_local, values = sample_gradient_on_line(
            gradient_img,
            pt1_local,
            pt2_local,
        )

        coords_global = [
            (x + x_offset, y + y_offset)
            for x, y in coords_local
        ]

        line_gradient = {
            "pt1": (pt1_local[0] + x_offset, pt1_local[1] + y_offset),
            "pt2": (pt2_local[0] + x_offset, pt2_local[1] + y_offset),
            "coords": coords_global,
            "values": values,
            "mean": float(np.mean(values)),
            "max": float(np.max(values)),
            "min": float(np.min(values)),
        }

    cut_mask = mask_uint8.copy()
    cv2.line(cut_mask, pt1_local, pt2_local, 0, thickness=2)

    return label(cut_mask > 0), deep_defects_global, line_gradient, hull_points_global

def get_and_split_all_labels(imgPath, dist, sigma_grad, depth, k=0.5):
    pic = EdgeFinder(imgPath, k=k)
    labels, markers = pic.watershed(min_distance=dist, sigma_grad=sigma_grad)

    # Reuse the gradient already created during watershed.
    grad_mag_smooth = pic.last_grad_mag_smooth

    final_labels = np.zeros_like(labels, dtype=np.int32)
    all_deep_defects = []
    all_line_gradients = []
    all_hulls = []
    current_label = 1

    # regionprops calculates bounding boxes once for all watershed labels.
    for prop in regionprops(labels):
        original_label = prop.label

        min_row, min_col, max_row, max_col = prop.bbox

        # Add a 1 px background border around the object - when bbox cropping too tight
        pad = 1

        row0 = max(0, min_row - pad)
        col0 = max(0, min_col - pad)
        row1 = min(labels.shape[0], max_row + pad)
        col1 = min(labels.shape[1], max_col + pad)

        local_label_roi = labels[row0:row1, col0:col1]
        local_gradient_roi = grad_mag_smooth[row0:row1, col0:col1]

        # Create the mask only in current cell bounding box
        one_blob_roi = local_label_roi == original_label

        # First original object is added for validation, when it splits, new parts are added here again
        regions_to_check = [one_blob_roi]

        # final_roi is a view to the current original object bounding box
        final_roi = final_labels[row0:row1, col0:col1]

        while regions_to_check:
            current_blob = regions_to_check.pop()

            split_labels, deep_defects, line_gradient, hull_points = split_touching_cells_by_defects(
                current_blob,
                depth_threshold=depth,
                gradient_img=local_gradient_roi,
                offset_xy=(col0, row0),
            )

            number_of_parts = int(split_labels.max())

            # Store diagnostics from every checked region
            all_deep_defects.extend(deep_defects)

            if line_gradient is not None:
                line_gradient["original_label"] = int(original_label)
                all_line_gradients.append(line_gradient)

            if hull_points is not None:
                all_hulls.append({
                    "original_label": int(original_label),
                    "hull_points": hull_points,
                })

            # Object successfully split into 2 or more parts
            if number_of_parts > 1:
                for part_label in range(1, number_of_parts + 1):
                    new_blob = split_labels == part_label
                    # Every new part will be validated again
                    regions_to_check.append(new_blob)

            else:
                # The object cannot be split further
                final_roi[current_blob] = current_label
                current_label += 1

    return final_labels, all_deep_defects, all_line_gradients, grad_mag_smooth, markers, all_hulls

# =========================== Profile analysis start - FOR FUTURE WORK ==============================
#def plot_line_gradient_profile(all_line_gradients):
#    fig, ax = plt.subplots(figsize=(7, 4))
#
#    for i, line_gradient in enumerate(all_line_gradients):
#        ax.plot(
#            line_gradient["values"],
#            label=f"{i}: {line_gradient['pt1']} -> {line_gradient['pt2']}"
#        )
#
#    ax.set_xlabel("Pixel position along line")
#    ax.set_ylabel("Gradient magnitude")
#    ax.set_title("Gradient profiles along cut lines")
#    ax.grid(True, alpha=0.3)
#    #ax.legend()
#    return fig
#
#

def analyze_line_gradients_global_min(all_line_gradients):
    results = []

    for i, line_gradient in enumerate(all_line_gradients):
        values = np.asarray(line_gradient["values"], dtype=float)

        if len(values) < 2:
            results.append({
                "line_index": i,
                "pt1": line_gradient["pt1"],
                "pt2": line_gradient["pt2"],
                "error": "Profile too short"
            })
            continue

        # first global maximum from the left
        max_idx = int(np.argmax(values))
        max_val = float(values[max_idx])

        # global minimum of the whole profile
        min_idx = int(np.argmin(values))
        min_val = float(values[min_idx])

        # analyze the segment between those two points
        left_idx = min(max_idx, min_idx)
        right_idx = max(max_idx, min_idx)
        segment = values[left_idx:right_idx + 1]

        diffs = np.diff(segment)

        if len(diffs) == 0:
            max_increase = 0.0
            max_increase_from = left_idx
            max_increase_to = left_idx
        else:
            best_diff_idx = int(np.argmax(diffs))
            max_increase = float(diffs[best_diff_idx])
            max_increase_from = left_idx + best_diff_idx
            max_increase_to = left_idx + best_diff_idx + 1

        total_change = float(min_val - max_val)
        total_drop = float(max_val - min_val)

        results.append({
            "line_index": i,
            "pt1": line_gradient["pt1"],
            "pt2": line_gradient["pt2"],
            "max_idx": max_idx,
            "max_val": max_val,
            "global_min_idx": min_idx,
            "global_min_val": min_val,
            "largest_increase_derivative": max_increase,
            "largest_increase_from_idx": max_increase_from,
            "largest_increase_to_idx": max_increase_to,
            "total_change_max_to_global_min": total_change,
            "total_drop_max_to_global_min": total_drop,
        })

    return results


# =========================== Profile analysis END ==============================

