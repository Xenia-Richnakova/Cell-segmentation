import numpy as np
import cv2
from scipy import ndimage as ndi
from skimage import exposure, filters, restoration, util

def to_float01(img: np.ndarray) -> np.ndarray:
    """Convert uint8/uint16/float images to float32 in [0, 1]."""
    img = np.asarray(img)
    if img.dtype.kind in ("u", "i"):
        info = np.iinfo(img.dtype)
        img = img.astype(np.float32) / float(info.max)
    else:
        img = img.astype(np.float32)
        # robust rescale if not already in [0,1]
        p1, p99 = np.percentile(img, [1, 99])
        if p99 > p1:
            img = (img - p1) / (p99 - p1)
        img = np.clip(img, 0.0, 1.0)
    return img

def illumination_correct_rolling_ball(img01: np.ndarray, radius: int = 64) -> np.ndarray:
    """
    Estimate background with rolling-ball and subtract.
    radius should be >= the typical radius of the largest 'non-background' structure.
    """
    bg = restoration.rolling_ball(img01, radius=radius)
    out = img01 - bg
    out = exposure.rescale_intensity(out, in_range="image", out_range=(0, 1))
    return out

def clahe_opencv(img01: np.ndarray, clip_limit: float = 2.0, tile_grid_size=(8, 8)) -> np.ndarray:
    """
    CLAHE via OpenCV expects 8-bit; convert and back.
    """
    img8 = np.clip(img01 * 255.0, 0, 255).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=float(clip_limit), tileGridSize=tuple(tile_grid_size))
    eq8 = clahe.apply(img8)
    return eq8.astype(np.float32) / 255.0

def preprocess_brightfield(
    img: np.ndarray,
    *,
    rolling_ball_radius: int = 64,
    gaussian_sigma: float = 1.0,
    use_clahe: bool = True,
    clahe_clip: float = 2.0,
    clahe_tile: tuple[int, int] = (8, 8),
    invert: bool = False,
) -> np.ndarray:
    """
    Canonical preprocessing:
      1) float [0,1]
      2) rolling-ball background subtraction
      3) mild gaussian denoise
      4) CLAHE (optional)
      5) optional inversion
    """
    img01 = to_float01(img)
    img01 = illumination_correct_rolling_ball(img01, radius=rolling_ball_radius)
    if gaussian_sigma and gaussian_sigma > 0:
        img01 = filters.gaussian(img01, sigma=gaussian_sigma, preserve_range=True).astype(np.float32)
    if use_clahe:
        img01 = clahe_opencv(img01, clip_limit=clahe_clip, tile_grid_size=clahe_tile)
    if invert:
        img01 = 1.0 - img01
    return np.clip(img01, 0.0, 1.0)
