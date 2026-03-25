import numpy as np
from scipy import ndimage as ndi
from scipy.ndimage import binary_fill_holes
from skimage import filters, measure, morphology, segmentation
from skimage.measure import regionprops

from preprocess_brightfield import preprocess_brightfield
from object_extractor import objectExtractor, select_the_most_regular


# ---------- helpers ----------

def gradually_select_best(seg, num_of_best=20):
    labels_copy = seg.labels.copy()
    props = regionprops(seg.labels)
    best_labels = []

    for _ in range(num_of_best):
        lbl = select_the_most_regular(props, labels_copy)
        if lbl is None:
            break
        best_labels.append(lbl)
        labels_copy[labels_copy == lbl] = 0

    return best_labels


def estimate_superpixels_from_cell_diameter(
    image_shape: tuple[int, int],
    cell_diameter_px: float | None,
    superpixels_per_cell: int = 20,
    default_n_segments: int = 500,
) -> int:
    H, W = image_shape
    if not cell_diameter_px or cell_diameter_px <= 0:
        return int(default_n_segments)

    cell_area = np.pi * (cell_diameter_px / 2.0) ** 2
    sp_area = max(cell_area / float(superpixels_per_cell), 9.0)
    n_segments = int((H * W) / sp_area)
    return int(np.clip(n_segments, 100, 20000))


def clean_binary_mask(mask: np.ndarray, *, min_obj: int = 200, min_hole: int = 200) -> np.ndarray:
    mask = mask.astype(bool)
    mask = binary_fill_holes(mask)
    mask = morphology.remove_small_objects(mask, min_size=min_obj)
    mask = morphology.remove_small_holes(mask, area_threshold=min_hole)
    mask = morphology.binary_closing(mask, morphology.disk(2))
    return mask


def superpixel_mean_image(img01: np.ndarray, sp_labels: np.ndarray) -> np.ndarray:
    labels = sp_labels.astype(np.int32)
    flat_l = labels.ravel()
    flat_x = img01.ravel()

    max_label = int(flat_l.max())
    sums = np.bincount(flat_l, weights=flat_x, minlength=max_label + 1)
    cnts = np.bincount(flat_l, minlength=max_label + 1)

    means = np.zeros_like(sums, dtype=np.float32)
    valid = cnts > 0
    means[valid] = (sums[valid] / cnts[valid]).astype(np.float32)

    out = means[labels]
    out[labels == 0] = 0.0
    return out


def feature_peak_local_max(dist: np.ndarray, mask: np.ndarray, min_distance: int) -> np.ndarray:
    from skimage.feature import peak_local_max
    return peak_local_max(
        dist,
        labels=mask.astype(np.uint8),
        min_distance=int(min_distance),
    )


def markers_from_distance(mask: np.ndarray, *, min_distance: int = 10, sigma: float = 1.0) -> np.ndarray:
    dist = ndi.distance_transform_edt(mask).astype(np.float32)

    if sigma and sigma > 0:
        dist = filters.gaussian(dist, sigma=sigma, preserve_range=True).astype(np.float32)

    coords = feature_peak_local_max(dist, mask, min_distance=min_distance)

    seed = np.zeros_like(mask, dtype=bool)
    if coords.size > 0:
        seed[coords[:, 0], coords[:, 1]] = True

    markers = measure.label(seed)
    return markers


def enforce_superpixel_consistency(labels: np.ndarray, sp_labels: np.ndarray) -> np.ndarray:
    out = labels.copy()

    for sp in np.unique(sp_labels):
        if sp == 0:
            continue

        m = sp_labels == sp
        vals, counts = np.unique(labels[m], return_counts=True)

        if len(vals) > 1 and vals[0] == 0:
            vals, counts = vals[1:], counts[1:]

        if len(vals) > 0:
            out[m] = vals[np.argmax(counts)]

    return out


def relabel_compact(labels: np.ndarray) -> np.ndarray:
    labels, _, _ = segmentation.relabel_sequential(labels)
    return labels


# ---------- new hybrid pipeline ----------

def segment_cells_slic_watershed(
    image_path,
    img,
    *,
    cell_diameter_px=None,
    invert=False,

    # preprocess_brightfield -> for grayscale image used by SLIC / gradient
    rolling_ball_radius=32,
    gaussian_sigma=1.0,
    use_clahe=False,

    # objectExtractor -> for foreground mask
    object_k=0.3,
    use_best_only=False,
    num_of_best=20,
    mask_min_obj=300,

    # SLIC
    compactness=1.0,
    slic_sigma=0.5,
    slic_zero=True,
    superpixels_per_cell=20,
    default_n_segments=1000,

    # markers / watershed
    marker_min_distance=40,
    marker_sigma=0.5,

    # optional postprocessing
    apply_superpixel_consistency=False,

    return_debug=False,
):
    """
    Hybrid pipeline:
      - img01 comes from preprocess_brightfield()
      - mask comes from objectExtractor()

    Returns:
      labels: instance segmentation labels, 0 = background
      debug: intermediates if return_debug=True
    """

    # 1) Preprocessed grayscale image for SLIC / gradient / watershed surface
    img01 = preprocess_brightfield(
        img,
        rolling_ball_radius=rolling_ball_radius,
        gaussian_sigma=gaussian_sigma,
        use_clahe=use_clahe,
        invert=invert,
    )

    # 2) Foreground mask from your objectExtractor
    seg = objectExtractor(
        image_path=image_path,
        image_czi=True,
        k=object_k,
    )

    if use_best_only:
        best_labels = gradually_select_best(seg, num_of_best=num_of_best)
        mask = np.isin(seg.labels, best_labels)
    else:
        mask = seg.labels > 0

    mask = clean_binary_mask(mask, min_obj=mask_min_obj, min_hole=mask_min_obj)

    # 3) Decide number of superpixels
    n_segments = estimate_superpixels_from_cell_diameter(
        img01.shape,
        cell_diameter_px,
        superpixels_per_cell=superpixels_per_cell,
        default_n_segments=default_n_segments,
    )

    # 4) SLIC only inside mask
    sp = segmentation.slic(
        img01,
        n_segments=int(n_segments),
        compactness=float(compactness),
        sigma=float(slic_sigma),
        slic_zero=bool(slic_zero),
        enforce_connectivity=True,
        start_label=1,
        mask=mask,
        channel_axis=None,
        convert2lab=False,
    )

    # 5) Replace each superpixel by its mean intensity
    sp_mean = superpixel_mean_image(img01, sp)

    # 6) Watershed topography
    grad = filters.sobel(sp_mean).astype(np.float32)

    # 7) Markers from distance transform of the mask
    markers = markers_from_distance(
        mask,
        min_distance=marker_min_distance,
        sigma=marker_sigma,
    )

    # 8) Watershed
    labels = segmentation.watershed(
        grad,
        markers=markers,
        mask=mask,
    )

    # 9) Optional superpixel snapping
    if apply_superpixel_consistency:
        labels = enforce_superpixel_consistency(labels, sp)

    # 10) Final cleanup
    labels[~mask] = 0
    labels = relabel_compact(labels).astype(np.int32)

    if return_debug:
        return labels, {
            "img01": img01,
            "mask": mask,
            "superpixels": sp,
            "sp_mean": sp_mean,
            "grad": grad,
            "markers": markers,
            "raw_object_labels": seg.labels,
        }

    return labels
