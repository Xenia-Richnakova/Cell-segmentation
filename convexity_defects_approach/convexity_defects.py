import cv2
from skimage.measure import label
import numpy as np
import matplotlib.pyplot as plt

from counting_cells import EdgeFinder
from skimage.draw import line as skline
from skimage.filters import gaussian

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

def split_touching_cells_by_defects(binary_mask, depth_threshold=5.0, gradient_img=None):
    binary_mask = binary_mask.astype(bool)
    mask_uint8 = (binary_mask.astype(np.uint8) * 255)

    contours, _ = cv2.findContours(
        mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return label(binary_mask), [], None

    cnt = max(contours, key=cv2.contourArea)

    hull = cv2.convexHull(cnt, returnPoints=False)
    if hull is None or len(hull) < 3:
        return label(binary_mask), [], None

    defects = cv2.convexityDefects(cnt, hull)
    if defects is None:
        return label(binary_mask), [], None

    deep_defects = []
    for i in range(defects.shape[0]):
        s, e, f, d = defects[i, 0]
        actual_depth = d / 256.0
        far_point = tuple(cnt[f][0])

        if actual_depth > depth_threshold:
            deep_defects.append((i, actual_depth, far_point))

    if len(deep_defects) < 2:
        return label(binary_mask), deep_defects, None

    deep_defects.sort(reverse=True, key=lambda x: x[1])

    idx1, depth1, pt1 = deep_defects[0]
    idx2, depth2, pt2 = deep_defects[1]

    print(f"Selected defect #{idx1} at {pt1} with depth {depth1:.2f}")
    print(f"Selected defect #{idx2} at {pt2} with depth {depth2:.2f}")

    line_gradient = None
    if gradient_img is not None:
        coords, values = sample_gradient_on_line(gradient_img, pt1, pt2)
        line_gradient = {
            "pt1": pt1,
            "pt2": pt2,
            "coords": coords,
            "values": values,
            "mean": float(np.mean(values)),
            "max": float(np.max(values)),
            "min": float(np.min(values)),
        }

    cut_mask = mask_uint8.copy()
    cv2.line(cut_mask, pt1, pt2, 0, thickness=2)

    return label(cut_mask > 0), deep_defects, line_gradient

def get_and_split_all_labels(imgPath, dist, sigma_grad, depth):
    pic = EdgeFinder(imgPath, k=0.5)
    labels, _ = pic.watershed(min_distance=dist, sigma_grad=sigma_grad)

    grad_mag = pic.gradient_magnitude()
    grad_mag_smooth = gaussian(grad_mag, sigma=sigma_grad)

    final_labels = np.zeros_like(labels, dtype=np.int32)
    all_deep_defects = []
    all_line_gradients = []
    current_label = 1

    for lbl in np.unique(labels):
        if lbl == 0:
            continue

        one_blob = labels == lbl
        split, deep_defects, line_gradient = split_touching_cells_by_defects(
            one_blob,
            depth_threshold=depth,
            gradient_img=grad_mag_smooth
        )

        split = label(split > 0)
        split[split > 0] += current_label - 1
        final_labels[split > 0] = split[split > 0]

        all_deep_defects.extend(deep_defects)

        if line_gradient is not None:
            line_gradient["original_label"] = int(lbl)
            all_line_gradients.append(line_gradient)

        current_label = final_labels.max() + 1

    return final_labels, all_deep_defects, all_line_gradients, grad_mag_smooth



from skimage.measure import regionprops

def plot_cells_w_numbers(labels, deep_defects):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(labels, cmap="nipy_spectral")

    # plot cell numbers
    """
    for region in regionprops(labels):
        y, x = region.centroid
        ax.text(
            x, y,
            str(region.label),
            color="white",
            fontsize=13,
            ha="center",
            va="center",
        )"""

    # plot deep defect numbers
    for defect_idx, depth, far_point in deep_defects:
        x, y = far_point
        ax.plot(x, y, "wo", markersize=4)
        ax.text(
            x + 5, y - 4,
            str(defect_idx),
            color="white",
            fontsize=10,
            ha="left",
            va="center",
        )

    ax.set_axis_off()
    return fig


def plot_gradient_heatmap_with_lines(grad_mag_smooth, line_gradients):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(grad_mag_smooth, cmap="inferno")

    for item in line_gradients:
        pt1 = item["pt1"]
        pt2 = item["pt2"]

        x1, y1 = pt1
        x2, y2 = pt2

        ax.plot([x1, x2], [y1, y2], color="cyan", linewidth=2)
        ax.plot(x1, y1, "wo", markersize=4)
        ax.plot(x2, y2, "wo", markersize=4)

        mx = (x1 + x2) / 2
        my = (y1 + y2) / 2
        ax.text(
            mx, my,
            f"{item['mean']:.2f}",
            color="white",
            fontsize=9,
            ha="center",
            va="center",
            bbox=dict(facecolor="black", alpha=0.5, pad=1),
        )

    ax.set_title("Gradient magnitude heat map with cut lines")
    ax.axis("off")
    return fig


def plot_line_gradient_profile(all_line_gradients):
    fig, ax = plt.subplots(figsize=(7, 4))

    for i, line_gradient in enumerate(all_line_gradients):
        ax.plot(
            line_gradient["values"],
            label=f"{i}: {line_gradient['pt1']} -> {line_gradient['pt2']}"
        )

    ax.set_xlabel("Pixel position along line")
    ax.set_ylabel("Gradient magnitude")
    ax.set_title("Gradient profiles along cut lines")
    ax.grid(True, alpha=0.3)
    ax.legend()
    return fig
