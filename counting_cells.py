from tifffile import imread
import matplotlib.pyplot as plt
from skimage import color
from scipy.ndimage import binary_fill_holes
import time
from object_extractor import objectExtractor, select_the_most_regular
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage.morphology import label
from skimage.segmentation import watershed
from skimage.measure import regionprops
import numpy as np
from skimage.filters import gaussian

path = "./moreCells.tif"



SX_X = [1, 0, -1]   # horizontal X
SX_Y = [
    [1],
    [2],
    [1]
]   # vertical X

SY_X = [1, 2, 1]   # horizontal Y
SY_Y = [
    [1],
    [0],
    [-1]
]   # vertical Y

def allocate_arr(H, W) -> list:
    return [[0.0 for _ in range(W)] for _ in range(H)]

class EdgeFinder:
    def __init__(self, img_path):
        self.path = img_path
        self.img = imread(self.path)
        self.img_grayscale = color.rgb2gray(self.img)

        seg = objectExtractor(image_path=self.path, )
        self.img_grayscale = seg.gray_smooth

        best_labels = self.gradually_select_best(seg)

        # Build one combined label‐mask
        selected = np.isin(seg.labels, best_labels).astype(int)

        self.seg = binary_fill_holes(selected)

        self.H, self.W = self.img_grayscale.shape

    def convolve(self, kernel, img, dir):
        out = allocate_arr(self.H, self.W)
        if dir == "x":
            for r in range(1, self.H - 1):
                for c in range(1, self.W - 1):
                    out[r][c] = (kernel[0] * img[r - 1][c] +
                                 kernel[1] * img[r][c] +
                                 kernel[2] * img[r + 1][c])

        if dir == "y":
            for r in range(1, self.H - 1):
                for c in range(1, self.W - 1):
                    out[r][c] = (kernel[0][0] * img[r][c - 1] +
                                 kernel[1][0] * img[r][c] +
                                 kernel[2][0] * img[r][c + 1])

        return out

    def gradient_x(self):
        tmp = self.convolve(SX_X, self.img_grayscale, "x")
        return self.convolve(SX_Y, tmp, "y")

    def gradient_y(self):
        tmp = self.convolve(SY_X, self.img_grayscale, "x")
        return self.convolve(SY_Y, tmp, "y")

    def gradient_magnitude(self):
        dx = ndi.sobel(self.img_grayscale, axis=1) # Horizontal
        dy = ndi.sobel(self.img_grayscale, axis=0) # Vertical

        # Calculate magnitude
        mag = np.hypot(dx, dy)

        return mag

    def gradually_select_best(self, seg, num_of_best=20):
        labels_copy = seg.labels.copy()
        props       = regionprops(seg.labels)
        best_labels = []
        for _ in range(num_of_best):
            lbl = select_the_most_regular(props, labels_copy)
            if lbl is None:
                break
            best_labels.append(lbl)
            labels_copy[labels_copy == lbl] = 0

        return best_labels

    def consider_largest_regions(self, props, min_area=4000):
        big_regions = []
        for p in props:
            if p.area >= min_area and p.perimeter > 1100:
                big_regions.append(p)

        return big_regions

    def plot_cells_w_numbers(self, labels, big_regions):
        fig, ax = plt.subplots(figsize=(6, 6))
        # Show the segmentation or grayscale
        ax.imshow(labels, cmap="nipy_spectral")
        # ax.imshow(pic.img_grayscale, cmap="gray")

        for i, region in enumerate(big_regions, start=1):
            y, x = region.centroid
            ax.text(
                x, y,
                str(i),
                color="white",
                fontsize=13,
                ha="center",
                va="center"
            )
        return fig

    def show_heat_map(self):
        mask = self.seg > 0
        distance = ndi.distance_transform_edt(mask).astype(float)
        fig, ax = plt.subplots(figsize=(6, 6))


        # Show the segmentation or grayscale
        ax.imshow(distance, cmap="inferno")
        plt.scatter(*self.coords.T[::-1])
        return fig

    def watershed(self, min_distance=20, sigma_dist=4.0):
        mask = self.seg > 0

        distance = ndi.distance_transform_edt(mask).astype(float)
        distance_smooth = gaussian(distance, sigma=sigma_dist)


        self.coords = peak_local_max(
            distance_smooth,
            labels=mask,
            min_distance=min_distance
        )

        seed_img = np.zeros_like(mask, dtype=bool)
        seed_img[tuple(self.coords.T)] = True
        markers = label(seed_img)

        mag = np.array(self.gradient_magnitude(), dtype=float)

        labels_ws = watershed(-distance, markers=markers, mask=mask, watershed_line=True)
        return labels_ws, markers




