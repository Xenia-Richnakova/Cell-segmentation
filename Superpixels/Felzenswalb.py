import numpy as np
from scipy import ndimage as ndi
from skimage import exposure, filters, measure, morphology, segmentation, util
from preprocess_brightfield import  preprocess_brightfield
from object_extractor import objectExtractor, select_the_most_regular
from skimage.measure import regionprops
from scipy.ndimage import binary_fill_holes
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

    for p in best_labels:
        print(p)
    return best_labels
# graph import compatible with scikit-image move from skimage.future.graph to skimage.graph
try:
    from skimage import graph
except Exception:  # older scikit-image
    from skimage.future import graph

def boundary_rag_merge(labels: np.ndarray, edge_map: np.ndarray, *, thresh: float = 0.05) -> np.ndarray:
    """
    Merge regions in `labels` using a boundary-weighted RAG.
    Edge weights are average edge strength along region boundaries.
    Merge until no edge weight < thresh.
    """
    rag = graph.rag_boundary(labels, edge_map, connectivity=2)

    # These functions follow the structure shown in scikit-image's boundary-merge example
    def _merge_boundary(g, src, dst):
        # merge src into dst: accumulate boundary "count" metadata (if present)
        for key in ("total boundary", "count"):
            if key not in g.nodes[dst]:
                g.nodes[dst][key] = 0.0
            if key not in g.nodes[src]:
                g.nodes[src][key] = 0.0
        g.nodes[dst]["total boundary"] += g.nodes[src]["total boundary"]
        g.nodes[dst]["count"] += g.nodes[src]["count"]

    def _weight_boundary(g, src, dst, n):
        # compute new boundary weight to neighbor n after merging src->dst
        w = 0.0
        c = 0.0
        for u in (src, dst):
            if g.has_edge(u, n):
                w += g.edges[u, n].get("weight", 0.0) * g.edges[u, n].get("count", 1.0)
                c += g.edges[u, n].get("count", 1.0)
        if c <= 0:
            return {"weight": 0.0, "count": 1.0}
        return {"weight": float(w / c), "count": float(c)}

    out = graph.merge_hierarchical(
        labels,
        rag,
        thresh=float(thresh),
        rag_copy=False,
        in_place_merge=True,
        merge_func=_merge_boundary,
        weight_func=_weight_boundary,
    )
    out, _, _ = segmentation.relabel_sequential(out)
    return out

def segment_cells_felzenszwalb_rag(
    img: np.ndarray,
    *,
    invert: bool = False,
    # preprocessing (same as pipeline A)
    rolling_ball_radius: int = 64,
    gaussian_sigma: float = 1.0,
    use_clahe: bool = True,
    # felzenszwalb
    fz_scale: float = 100.0,
    fz_sigma: float = 0.8,
    fz_min_size: int = 20,
    # rag merge
    rag_thresh: float = 0.05,
    # cell mask & splitting
    make_instance_labels: bool = True,
    marker_min_distance: int = 10,
    marker_sigma: float = 1.0,
    min_object_area: int = 300,
    return_debug: bool = False,
):
    """
    Pipeline:
      preprocess -> felzenszwalb superpixels -> boundary-RAG merge -> (optional) watershed split -> clean/relabel
    """
    img01 = preprocess_brightfield(
        img,
        rolling_ball_radius=rolling_ball_radius,
        gaussian_sigma=gaussian_sigma,
        use_clahe=use_clahe,
        invert=invert,
    )

    # Edge map for boundary evidence
    edge = filters.sobel(img01).astype(np.float32)

    # Felzenszwalb on grayscale
    sp0 = segmentation.felzenszwalb(
        img01,
        scale=float(fz_scale),
        sigma=float(fz_sigma),
        min_size=int(fz_min_size),
        channel_axis=None,
    )
    sp0, _, _ = segmentation.relabel_sequential(sp0)

    # Merge superpixels where boundary evidence is weak
    sp = boundary_rag_merge(sp0, edge_map=edge, thresh=rag_thresh)

    # Build a conservative foreground mask from merged regions:
    # Use Otsu on the preprocessed intensity as a generic default.
    thr = filters.threshold_otsu(img01)
    fg = img01 > thr
    fg = morphology.remove_small_objects(fg, min_size=min_object_area)
    fg = morphology.remove_small_holes(fg, area_threshold=min_object_area)
    fg = morphology.binary_closing(fg, morphology.disk(2))

    if not make_instance_labels:
        # Return merged superpixels restricted to foreground as a pseudo-instance map.
        labels = sp.copy()
        labels[~fg] = 0
        labels, _, _ = segmentation.relabel_sequential(labels)
        if return_debug:
            return labels.astype(np.int32), {"img01": img01, "edge": edge, "sp0": sp0, "sp": sp, "fg": fg}
        return labels.astype(np.int32)

    # Otherwise: split with watershed using distance markers (as in pipeline A)
    dist = ndi.distance_transform_edt(fg).astype(np.float32)
    if marker_sigma and marker_sigma > 0:
        dist = filters.gaussian(dist, sigma=marker_sigma, preserve_range=True).astype(np.float32)

    from skimage.feature import peak_local_max
    coords = peak_local_max(dist, labels=fg.astype(np.uint8), min_distance=int(marker_min_distance))
    seed = np.zeros_like(fg, dtype=bool)
    if coords.size > 0:
        seed[coords[:, 0], coords[:, 1]] = True
    markers = measure.label(seed)

    # Use edge map as watershed surface
    labels = segmentation.watershed(edge, markers=markers, mask=fg)
    labels, _, _ = segmentation.relabel_sequential(labels)

    if return_debug:
        return labels.astype(np.int32), {"img01": img01, "edge": edge, "sp0": sp0, "sp": sp, "fg": fg, "markers": markers}
    return labels.astype(np.int32)
