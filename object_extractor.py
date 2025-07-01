import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color, filters, morphology, util, exposure
from scipy import ndimage as ndi
from skimage.measure import regionprops
from skimage.measure._regionprops import RegionProperties
from scipy.ndimage import binary_fill_holes
from skimage.segmentation import find_boundaries
from skimage.morphology import binary_dilation, disk



def select_the_most_regular(props, labels, min_area_convex=15000):
    props = regionprops(labels)
    # only keep regions big enough
    valid = [p for p in props if p.area_convex > min_area_convex]
    if not valid:
        return None     # nothing left

    # initialize on the first valid region
    best = valid[0]
    best_euler = abs(best.euler_number)

    # 1) pick smallest euler among valid
    for obj in valid:
        if abs(obj.euler_number) < best_euler:
            best_euler = abs(obj.euler_number)
            best = obj

    # 2) if tie on euler, pick the one with larger convex area
    for obj in valid:
        if abs(obj.euler_number) == best_euler and obj.area_convex > best.area_convex:
            best = obj

    return best.label


class objectExtractor:
    def __init__(self, image_path=None, image_array=None,noise_suppression_var=0.05, sigma_value=1.6, k=0.2):
        self.k = k
        self.counter = 0

        # --- load image ---
        if image_array is not None:
            img = image_array.copy()
        else:
            img = io.imread(image_path)

        self.original = img

        # now img is a float array in [0,1]
        if img.dtype != np.float32 and img.dtype != np.float64:
            img = util.img_as_float(img)

        img = (img - img.min()) / (img.max() - img.min())

        # if 3‐channel, convert to gray; if already 2D, leave as is
        if img.ndim == 3 and img.shape[2] in (3,4):
            # RGB or RGBA
            self.gray = color.rgb2gray(img)
        else:
            # already single‐channel
            self.gray = img

        # Gaussian filter for background noise suppression
        self.gray_smooth = filters.gaussian(self.gray, sigma=sigma_value)
        self.grey_smooth = exposure.rescale_intensity(self.gray_smooth, out_range=(0,1))

        # Niblack thresholding
        self.thresh = filters.threshold_niblack(self.gray_smooth, k=self.k)
        #self.thresh = filters.threshold_otsu(self.gray_smooth)

        self.binary_global = None
        result = self.make_binary(noise_suppression_var)
        if result is None:
            self.props, self.labels = [], np.zeros_like(self.binary_global, dtype=int)
            return
        else:
            self.props, self.labels = result

        # check and adjust largest object regularity, if euler not equal to one, lower noise suppression variable
        self.candidate = largest = max(self.props, key=lambda p: p.area).label
        object: RegionProperties = self.adjust_largest_object_regularity(self.props, self.labels, noise_suppression_var, largest)
        if object.euler_number < -8:
            self.props, self.labels = self.make_binary(noise_suppression_var)
            best = select_the_most_regular(self.props, self.labels)
            self.candidate = best
            self.adjust_largest_object_regularity(self.props, self.labels, noise_suppression_var, best)

    def adjust_largest_object_regularity(self, props, labels, noise_suppression_var, canditate) -> RegionProperties:
        props = regionprops(labels)
        if canditate is None:
            return None
        p = next(p for p in props if p.label == canditate)
        best_p = p
        best_euler = abs(p.euler_number)

        print(f"Area Convex     : {p.area_convex:.2f}")
        print(f"Euler           : {p.euler_number:.2f}")

        previous_euler = p.euler_number

        while (p.euler_number != 0) and self.check_if_better(previous_euler, p.euler_number) and noise_suppression_var > 0.005:
            previous_euler = p.euler_number
            noise_suppression_var -= 0.005
            props, labels = self.make_binary(noise_suppression_var)
            largest = max(props, key=lambda p: p.area).label
            props = regionprops(labels)
            p = next(p for p in props if p.label == largest)

            if p.euler_number < best_euler:
                best_euler = p.euler_number
                best_p = p

            print(f"Area Convex     : {p.area_convex:.2f}")
            print(f"Euler           : {p.euler_number:.2f}  ")

        #self.props = props
        self.props = regionprops(labels)    # re-label based on best_p
        self.labels = labels                # keep the labels from best_p’s iteration

        return best_p


    def check_if_better(self, previous_euler, current_euler):
        if previous_euler < 0:
            if previous_euler > current_euler:
                return False
        if previous_euler > 0:
            if previous_euler < current_euler:
                return False

        return  True

    def get_all_objects(self):
        return self.props

    def make_binary(self, noise_suppression_var):
        # 1) Threshold to binary
        self.binary_global = self.gray_smooth - noise_suppression_var > self.thresh

        # 2) Remove small objects (noise) and fill small holes
        cleaned = morphology.remove_small_objects(self.binary_global, min_size=500)
        cleaned = morphology.remove_small_holes(cleaned, area_threshold=500)

        # 3) Further clean-up: close narrow gaps
        selem = disk(3)
        cleaned = morphology.binary_closing(cleaned, selem)

        # 4) A little dilation to thicken the mask
        selem2 = disk(1)
        cleaned = morphology.binary_dilation(cleaned, selem2)

        # 5) Label and compute regionprops
        labels, num = ndi.label(cleaned)
        props = regionprops(labels)

        # 6) fallback if no props
        if len(props) == 0 and self.counter < 5:
            self.counter += 1
            return self.make_binary(noise_suppression_var - 0.005)
        if len(props) == 0 and self.counter >= 5:
            print("No object was detected")
            return None

        return props, labels

    def plot_results(self, image_name = '', n_best=10, do_plot=False):
        # 1) select the top labels
        labels_copy = self.labels.copy()
        props       = regionprops(self.labels)
        best_labels = []
        for _ in range(n_best):
            lbl = select_the_most_regular(props, labels_copy)
            if lbl is None:
                break
            best_labels.append(lbl)
            labels_copy[labels_copy == lbl] = 0

        # 2) build one combined label‐mask
        selected = np.isin(self.labels, best_labels).astype(int)

        # fill all selected regions (won’t merge, since label regions are separate)
        filled = binary_fill_holes(selected)

        # find pixel‐wide boundaries between labels
        bnd = find_boundaries(selected, mode='inner')
        # make them a little thicker
        bnd = binary_dilation(bnd, disk(1))

        fig, axes = plt.subplots(1, 3, figsize=(12,4))
        for ax in axes:
            ax.axis('off')

        axes[0].imshow(self.original, cmap='gray')
        axes[0].set_title(f'Original image {image_name}')

        #axes[1].imshow(self.labels>0, cmap='gray')
        #axes[1].set_title(f"Top {len(best_labels)} regions")
        axes[1].imshow(filled, cmap='gray', vmin=0, vmax=1)
        axes[1].set_facecolor('black')

        # Panel 3: black bg, white fill
        axes[2].imshow(filled, cmap='gray', vmin=0, vmax=1)
        axes[2].set_facecolor('black')

        # contour at 0.5 boundary
        axes[2].contour(
            bnd,               # or combined_mask_filled
            levels=[0.5],
            colors='red',
            linewidths=1
        )

        plt.tight_layout()
        if do_plot:
            plt.show()
        return fig



