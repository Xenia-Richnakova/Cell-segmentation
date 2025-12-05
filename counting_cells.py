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


path = "./1355-8xK_gal_4822.tif"
#path = "./moreCells.tif"


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

        seg = objectExtractor(image_path=self.path)
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
        # |magn| = sqrt(dx^2 + dy^2)
        mag = allocate_arr(self.H, self.W)

        dx = self.gradient_x()
        dy = self.gradient_y()
        for y in range(self.H):
            for x in range(self.W):
                mag[y][x] = dx[y][x] ** 2 + dy[y][x] ** 2
                #mag[y][x] = math.sqrt(dx[y][x] ** 2 + dy[y][x] ** 2) * 2

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

    def consider_largest_regions(self, props):
        big_regions = []
        for p in props:
            if p.area >= 4000 and p.perimeter > 1100:
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

        ax.axis("off")
        plt.tight_layout()
        plt.show()


    def watershed(self, min_distance=20, sigma_dist=4.0):
        mask = self.seg > 0

        distance = ndi.distance_transform_edt(mask).astype(float)
        distance_smooth = gaussian(distance, sigma=sigma_dist)

        coords = peak_local_max(
            distance_smooth,
            labels=mask,
            min_distance=min_distance
        )

        seed_img = np.zeros_like(mask, dtype=bool)
        seed_img[tuple(coords.T)] = True
        markers = label(seed_img)

        mag = np.array(self.gradient_magnitude(), dtype=float)

        labels_ws = watershed(mag, markers=markers, mask=mask)
        return labels_ws, markers


    def show(self, image=None):
        data = image if image is not None else self.img_grayscale
        plt.imshow(data, cmap="gray", vmin=0.0, vmax=1.0)
        plt.axis("off")
        plt.show()


start = time.time()

pic = EdgeFinder(path)
labels, markers = pic.watershed(min_distance=2, sigma_dist=5.0)

# Measure watershed regions
props = regionprops(labels)
areas = np.array([p.area for p in props])

# --- Area distribution ---
print("Number of watershed regions:", len(props))
print("Smallest / largest area:", areas.min(), areas.max())

big_regions = pic.consider_largest_regions(props)
print("Number of BIG regions (cells):", len(big_regions))

pic.plot_cells_w_numbers(labels, big_regions)

end = time.time()
print("Elapsed:", end - start, "seconds")

