import numpy as np
from skimage import io, color, filters, morphology
from scipy import ndimage as ndi
from skimage.measure import regionprops
from skimage.measure._regionprops import RegionProperties
from skimage.morphology import disk
from czifile import CziFile



def select_the_most_regular(props, labels=None, min_area_convex=15000):
    valid = [p for p in props if p.area_convex > min_area_convex]
    if not valid:
        return None

    # 1 lowest absolute Euler number
    # 2 largest convex area in case of a tie
    best = min(valid, key=lambda prop: (
            abs(prop.euler_number), -prop.area_convex),
    )

    return best.label


class objectExtractor:
    def __init__(self, image_path=None, image_czi=False, noise_suppression_var=0.05, sigma_value=1.6, k=0.3):
        self.k = k
        self.counter = 0

        # load image
        if image_czi:
            czi = CziFile(image_path)
            data = czi.asarray()
            img = np.squeeze(data)

            # if still not 2D, reduce
            while img.ndim > 2:
                img = img[0]

            if img.ndim != 2:
                raise ValueError(f"Expected 2D image after squeezing, got shape {img.shape}")
        else:
            img = io.imread(image_path)

        # normalize
        img = (img - img.min()) / (img.max() - img.min())

        self.original = img
        # if 3‐channel, convert to gray, if already 2D dont change
        if img.ndim == 3 and img.shape[2] in (3,4):
            # RGB or RGBA
            self.gray = color.rgb2gray(img)
        else:
            # already single‐channel
            self.gray = img

        self.gray = img
        # Gaussian filter for background noise suppression
        self.gray_smooth = filters.gaussian(self.gray, sigma=sigma_value)

        # Niblack thresholding
        self.thresh = filters.threshold_niblack(self.gray_smooth, k=self.k)

        self.binary_global = None
        result = self.make_binary(noise_suppression_var)
        if result is None:
            self.props, self.labels = [], np.zeros_like(self.binary_global, dtype=int)
            return
        else:
            self.props, self.labels = result

        # check and adjust largest object regularity, if euler not equal to 0, lower noise suppression variable
        self.candidate = largest = max(self.props, key=lambda p: p.area).label
        object: RegionProperties = self.adjust_largest_object_regularity(self.props, self.labels, noise_suppression_var, largest)
        if object.euler_number < -8:
            self.props, self.labels = self.make_binary(noise_suppression_var)
            best = select_the_most_regular(self.props, self.labels)
            self.candidate = best
            self.adjust_largest_object_regularity(self.props, self.labels, noise_suppression_var, best)

    def adjust_largest_object_regularity(self, props, labels, noise_suppression_var, canditate) -> RegionProperties:
        if canditate is None or not props:
            return None

        props_by_label = {
            prop.label: prop
            for prop in props
        }

        p = props_by_label.get(canditate)

        if p is None:
            return None

        best_p = p
        best_euler = abs(p.euler_number)
        previous_euler = p.euler_number

        current_props = props
        current_labels = labels

        while (p.euler_number != 0 and self.check_if_better(previous_euler, p.euler_number) and noise_suppression_var > 0.005):
            previous_euler = p.euler_number
            noise_suppression_var -= 0.005

            result = self.make_binary(noise_suppression_var)
            if result is None:
                break

            current_props, current_labels = result
            largest_label = max(current_props, key=lambda prop: prop.area).label

            props_by_label = {
                prop.label: prop
                for prop in current_props
            }

            p = props_by_label.get(largest_label)
            if p is None:
                break

            if p.euler_number < best_euler:
                best_euler = p.euler_number
                best_p = p

        self.props = current_props
        self.labels = current_labels

        return best_p


    def check_if_better(self, previous_euler, current_euler):
        if previous_euler < 0:
            if previous_euler > current_euler:
                return False
        if previous_euler > 0:
            if previous_euler < current_euler:
                return False

        return True

    def make_binary(self, noise_suppression_var):
        # Threshold to binary
        self.binary_global = self.gray_smooth - noise_suppression_var > self.thresh

        # Remove small objects and fill small holes
        cleaned = morphology.remove_small_objects(self.binary_global, min_size=500)
        cleaned = morphology.remove_small_holes(cleaned, area_threshold=500)

        # Close narrow gaps
        selem = disk(3)
        cleaned = morphology.closing(cleaned, selem)

        # Dilation to thicken the mask
        selem2 = disk(1)
        cleaned = morphology.dilation(cleaned, selem2)

        # Label and compute regionprops
        labels, num = ndi.label(cleaned)
        props = regionprops(labels)

        # fallback if no props
        if len(props) == 0 and self.counter < 5:
            self.counter += 1
            return self.make_binary(noise_suppression_var - 0.005)
        if len(props) == 0 and self.counter >= 5:
            print("No object was detected")
            return None


        return props, labels


