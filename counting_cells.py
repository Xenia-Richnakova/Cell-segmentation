from tifffile import imread
import matplotlib.pyplot as plt
from scipy.ndimage import binary_fill_holes
from object_extractor import objectExtractor, select_the_most_regular
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage.morphology import label
from skimage.segmentation import watershed
from skimage.measure import regionprops
import numpy as np
from skimage.filters import gaussian
from czifile import CziFile
from skimage.segmentation import find_boundaries
from PIL import Image


def allocate_arr(H, W) -> list:
    return [[0.0 for _ in range(W)] for _ in range(H)]


class EdgeFinder:
    def __init__(self, img_path):
        self.path = img_path

        if ".czi" in self.path:
            self.img = CziFile(self.path).asarray()
            avg_brightness = float(np.mean(self.img))
            print("Average brightness:", avg_brightness)
            arr = np.squeeze(self.img)   # remove singleton dimensions
            self.img = arr


            plt.figure(figsize=(6, 6))
            plt.imshow(self.img, cmap="gray")
            plt.axis("off")
            plt.show()

            seg = objectExtractor(image_path=self.path, image_czi=True)
        else:
            self.img = imread(self.path)
            seg = objectExtractor(image_path=self.path)


        #seg = objectExtractor(image_path=self.path)
        self.img_grayscale = seg.gray_smooth

        best_labels = self.gradually_select_best(seg)

        # Build one combined label‐mask
        selected = np.isin(seg.labels, best_labels).astype(int)
        self.seg = binary_fill_holes(selected)

        self.H, self.W = self.img_grayscale.shape
        self.coords = None

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

    def gradient_magnitude(self):
        dx = ndi.sobel(self.img_grayscale, axis=1)  # Horizontal
        dy = ndi.sobel(self.img_grayscale, axis=0)  # Vertical
        mag = np.hypot(dx, dy)
        return mag

    def gradually_select_best(self, seg, num_of_best=20):
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

    def consider_largest_regions(self, props, min_area=4000):
        big_regions = []
        for p in props:
            if p.area >= min_area and p.perimeter > 1100:
                big_regions.append(p)
        return big_regions

    def plot_cells_w_numbers(self, labels, big_regions):
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(labels, cmap="nipy_spectral")
        for i, region in enumerate(big_regions, start=1):
            y, x = region.centroid
            ax.text(x, y, str(i), color="white", fontsize=13, ha="center", va="center")
        return fig

    def show_heat_map(self):
        mask = self.seg > 0
        distance = ndi.distance_transform_edt(mask).astype(float)
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(distance, cmap="inferno")
        if self.coords is not None and len(self.coords) > 0:
            plt.scatter(*self.coords.T[::-1])
        return fig

    def plot_ws_overlay(self, labels_ws):
        boundaries = find_boundaries(labels_ws, mode="outer")
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(self.img_grayscale, cmap="gray")
        ax.imshow(boundaries, cmap="Reds", alpha=0.6)

        regions = regionprops(labels_ws)
        for i, region in enumerate(regions, start=1):
            y, x = region.centroid
            ax.text(x, y, str(i), color="yellow", fontsize=12, weight="bold",
                    ha="center", va="center")

        ax.axis("off")
        return fig

    # ----------------------------
    # NEW: proper method version
    # ----------------------------
    def watershed(self, min_distance=20, sigma_dist=4.0, sigma_grad=1.0):
        mask = self.seg > 0

        distance = ndi.distance_transform_edt(mask).astype(float)
        distance_smooth = gaussian(distance, sigma=sigma_dist)

        self.coords = peak_local_max(
            distance_smooth,
            labels=mask,
            min_distance=min_distance
        )

        seed_img = np.zeros_like(mask, dtype=bool)
        if self.coords is not None and len(self.coords) > 0:
            seed_img[tuple(self.coords.T)] = True
        markers = label(seed_img)

        mag = self.gradient_magnitude()
        mag_smooth = gaussian(mag, sigma=sigma_grad)

        #labels_ws = watershed(
        #    mag_smooth,
        #    markers=markers,
        #    mask=mask,
        #    watershed_line=True
        #)
        labels_ws = watershed(-distance_smooth, markers, mask=mask)

        return labels_ws, markers

    # ----------------------------
    # NEW: 0 background, cells: 255, 254, ...
    # ----------------------------
    @staticmethod
    def labels_to_uint8_reverse(labels_ws: np.ndarray) -> np.ndarray:
        """
        Strict mapping:
          background (0) -> 0
          1 -> 255
          2 -> 254
          ...
          255 -> 1

        Assumption:
          labels_ws contains at most 255 non-zero labels.
          If this assumption is violated, an exception is raised.
        """
        labels_ws = np.asarray(labels_ws)

        max_label = int(labels_ws.max())
        if max_label > 255:
            raise ValueError(
                f"labels_to_uint8_reverse(): found label {max_label}, "
                "but max allowed is 255 for 8-bit PNG export."
            )

        out = np.zeros(labels_ws.shape, dtype=np.uint8)

        mask = labels_ws > 0
        out[mask] = (256 - labels_ws[mask]).astype(np.uint8)

        return out

    # ----------------------------
    # NEW: save PNG
    # ----------------------------
    def save_label_png(self, labels_ws: np.ndarray, out_path: str) -> str | None:
        mapped, warning = self.labels_to_uint8_reverse(labels_ws)
        img = Image.fromarray(mapped, mode="L")  # 8-bit grayscale
        img.save(out_path)
        return warning

#path = "./Snap-7253.czi"
#path = "./moreCells.tif"
#pic = EdgeFinder(path)
