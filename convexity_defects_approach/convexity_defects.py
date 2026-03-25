import cv2
from skimage.measure import label
import numpy as np
import matplotlib.pyplot as plt

from counting_cells import EdgeFinder

def split_touching_cells_by_defects(binary_mask, depth_threshold=5.0):
    binary_mask = binary_mask.astype(bool)
    mask_uint8 = (binary_mask.astype(np.uint8) * 255)

    contours, _ = cv2.findContours(
        mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return label(binary_mask), []

    cnt = max(contours, key=cv2.contourArea)

    hull = cv2.convexHull(cnt, returnPoints=False)
    if hull is None or len(hull) < 3:
        return label(binary_mask), []

    defects = cv2.convexityDefects(cnt, hull)
    if defects is None:
        return label(binary_mask), []

    deep_defects = []
    for i in range(defects.shape[0]):
        s, e, f, d = defects[i, 0]
        actual_depth = d / 256.0
        far_point = tuple(cnt[f][0])

        #print(f"defect #{i}: start={s}, end={e}, far={f}, "f"depth={actual_depth:.2f}, far_point={far_point}" )

        if actual_depth > depth_threshold:
            deep_defects.append((i, actual_depth, far_point))

    if len(deep_defects) < 2:
        return label(binary_mask), deep_defects

    deep_defects.sort(reverse=True, key=lambda x: x[1])

    idx1, depth1, pt1 = deep_defects[0]
    idx2, depth2, pt2 = deep_defects[1]

    print(f"Selected defect #{idx1} at {pt1} with depth {depth1:.2f}")
    print(f"Selected defect #{idx2} at {pt2} with depth {depth2:.2f}")

    cut_mask = mask_uint8.copy()
    cv2.line(cut_mask, pt1, pt2, 0, thickness=2)

    return label(cut_mask > 0), deep_defects

def get_and_split_all_labels(imgPath, dist, sigma_grad, depth):
    pic = EdgeFinder(imgPath)
    labels, _ = pic.watershed(min_distance=dist, sigma_grad=sigma_grad)

    final_labels = np.zeros_like(labels, dtype=np.int32)
    all_deep_defects = []
    current_label = 1

    for lbl in np.unique(labels):
        if lbl == 0:
            continue

        one_blob = labels == lbl
        split, deep_defects = split_touching_cells_by_defects(one_blob, depth)

        split = label(split > 0)
        split[split > 0] += current_label - 1
        final_labels[split > 0] = split[split > 0]

        all_deep_defects.extend(deep_defects)
        current_label = final_labels.max() + 1

    return final_labels, all_deep_defects


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

