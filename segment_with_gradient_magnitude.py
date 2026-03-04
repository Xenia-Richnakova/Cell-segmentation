import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color, filters, morphology, util, exposure
from scipy import ndimage as ndi
from skimage.measure import regionprops
from skimage.measure._regionprops import RegionProperties
from scipy.ndimage import binary_fill_holes
from skimage.segmentation import find_boundaries
from skimage.morphology import binary_dilation, disk
from czifile import CziFile
from skimage.filters import gaussian
from object_extractor import select_the_most_regular

# Pridat do bakalarky ze skusala som segmentvat aj cez Gradient Magnitude ale nie je to lepsie

class Segment_With_Magnitude:
    def __init__(self, image_path=None, image_czi=False, k=0.3):
        self.k = k
        self.counter = 0

        # --- load image ---
        if image_czi:
            czi  = CziFile(image_path)
            data = czi.asarray()
            img = data[0, :, :, 0]
        else:
            img = io.imread(image_path)

        img = (img - img.min()) / (img.max() - img.min())
        self.image = img

        if img.ndim == 3 and img.shape[2] in (3,4):
            # RGB or RGBA
            self.gray = color.rgb2gray(img)
        else:
            # already single‐channel
            self.gray = img


    def gradient_magnitude(self):
        dx = ndi.sobel(self.gray, axis=1)  # Horizontal
        dy = ndi.sobel(self.gray, axis=0)  # Vertical
        mag = np.hypot(dx, dy)
        return mag

    def gradually_select_best(self, labels, num_of_best=20):
        labels_copy = labels.copy()
        best_labels = []

        for _ in range(num_of_best):
            lbl = select_the_most_regular(None, labels_copy)  # props arg unused anyway
            if lbl is None:
                break
            best_labels.append(lbl)
            labels_copy[labels_copy == lbl] = 0

        return best_labels

    def show_gradient_map(self, mag_smooth):
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(mag_smooth, cmap="inferno")
        return fig


    def consider_largest_regions(self, props, min_area=1000):
        big_regions = []
        for p in props:
            if p.area >= min_area and p.perimeter > 1100:
                big_regions.append(p)
        return big_regions

    """
    def segment(self, sigma_grad=10):
        from skimage.morphology import reconstruction
        mag = self.gradient_magnitude()
        mag_smooth = gaussian(mag, sigma=sigma_grad)

        # 1. Define a 'seed' and a 'mask'
        # We want to fill areas that are dark. We invert the image so that
        # the interiors become bright (easy to fill) and the edges remain high-intensity barriers.
        seed = np.copy(mag_smooth)
        seed[1:-1, 1:-1] = mag_smooth.max()
        mask = mag_smooth

        # 2. Perform reconstruction by erosion
        # This 'fills' the dark areas of the image up to the boundary
        # of the high-intensity edges (the cell walls).
        filled_img = reconstruction(seed, mask, method='erosion')

        # 3. Create binary mask
        # Now that the interiors are 'flat' (filled), we can easily
        # separate the object from the background.
        binary_filled = filled_img < (filled_img.max() * 0.2)

        return binary_filled

        #return mag_smooth
        """

    def segment(self, sigma_grad=2):
        # 1. Get the gradient magnitude (the 'rings')
        mag = self.gradient_magnitude()
        mag_smooth = gaussian(mag, sigma=sigma_grad)

        # 2. Identify the 'bright' pixels that form the enclosure.
        # We use a simple local comparison to create a mask of the borders.
        mask = mag_smooth > (mag_smooth.mean() + mag_smooth.std())

        # 3. Fill the holes enclosed by these borders
        # This operator treats the 'True' pixels as a boundary and
        # fills everything contained within them.
        filled = ndi.binary_fill_holes(mask)

        return filled
