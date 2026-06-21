from tifffile import imread
#import matplotlib.pyplot as plt
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
from PIL import Image



class EdgeFinder:
    def __init__(self, img_path, k=0.2):
        self.path = img_path

        if ".czi" in self.path:
            self.img = CziFile(self.path).asarray()
            avg_brightness = float(np.mean(self.img))
            print("Image:", self.path)
            print("Average brightness:", avg_brightness)
            arr = np.squeeze(self.img)   # remove singleton dimensions
            self.img = arr

            #plt.figure(figsize=(6, 6))
            #plt.imshow(self.img, cmap="gray")
            #plt.axis("off")
            #plt.show()

            seg = objectExtractor(image_path=self.path, image_czi=True, k=k)
        else:
            self.img = imread(self.path)
            seg = objectExtractor(image_path=self.path)

        self.img_grayscale = seg.gray_smooth

        best_labels = self.gradually_select_best(seg)

        # Build one combined label‐mask
        selected = np.isin(seg.labels, best_labels).astype(int)
        self.seg = binary_fill_holes(selected)

        self.H, self.W = self.img_grayscale.shape
        self.coords = None

        self._gradient_mag_cache = None
        self._smoothed_gradient_cache = {}
        self.last_grad_mag_smooth = None



    def gradient_magnitude(self):
        if self._gradient_mag_cache is None:
            dx = ndi.sobel(self.img_grayscale, axis=1)
            dy = ndi.sobel(self.img_grayscale, axis=0)

            self._gradient_mag_cache = np.hypot(dx, dy)

        return self._gradient_mag_cache

    def smoothed_gradient(self, sigma_grad):
        sigma_grad = float(sigma_grad)

        if sigma_grad not in self._smoothed_gradient_cache:
            self._smoothed_gradient_cache[sigma_grad] = gaussian(self.gradient_magnitude(), sigma=sigma_grad)

        return self._smoothed_gradient_cache[sigma_grad]

    def gradually_select_best(self, seg, num_of_best=20):
        valid_props = [prop for prop in seg.props if prop.area_convex > 15000]
        valid_props.sort(key=lambda prop: (abs(prop.euler_number), -prop.area_convex))

        best_labels = []
        for prop in valid_props[:num_of_best]:
            best_labels.append(prop.label)

        return best_labels

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

        mag_smooth = self.smoothed_gradient(sigma_grad)
        self.last_grad_mag_smooth = mag_smooth

        # Potential Image
        pot = mag_smooth * (1 - (distance_smooth / (distance_smooth.max() + 1e-9)))
        labels_ws = watershed(pot, markers, mask=mask)

        return labels_ws, markers



