import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color, filters, morphology, util, exposure
from scipy import ndimage as ndi
from skimage.measure import regionprops
from skimage.measure._regionprops import RegionProperties




def select_the_most_regular(props, labels):
    props = regionprops(labels)
    best_euler= abs(props[0].euler_number)
    best = props[0]
    for obj in props:
        print(f"{abs(obj.euler_number)} < {best_euler}")

        if abs(obj.euler_number) < best_euler and obj.area > 100:
            best_euler = abs(obj.euler_number)
            best = obj

    print(best, best_euler)
    for o in props:
        o_perim_ratio = o.area / o.perimeter
        best_perim_ratio = best.area / best.perimeter
        if o.euler_number == best_euler and o_perim_ratio < best_perim_ratio:
            best = o
    return best.label

class objectExtractor:
    def __init__(self, image_path, noise_suppression_var=0.05):
        self.counter = 0
        # Load the image
        img = io.imread(image_path)
        # Convert to grayscale
        self.gray = color.rgb2gray(img)
        # Gaussian filter for background noise suppression
        self.gray_smooth = filters.gaussian(self.gray, sigma=1.1)
        # Otsu thresholding
        self.thresh = filters.threshold_otsu(self.gray_smooth)
        self.binary_global = None
        self.props, self.labels = self.make_binary(noise_suppression_var)

        # check and adjust largest object regularity, if euler not equal to one, lower noise suppression variable
        self.candidate = largest = max(self.props, key=lambda p: p.area).label
        object: RegionProperties = self.adjust_largest_object_regularity(self.props, self.labels, noise_suppression_var, largest)
        if object.euler_number < -8:
            print("Second round")
            self.props, self.labels = self.make_binary(noise_suppression_var)
            best = select_the_most_regular(self.props, self.labels)
            self.candidate = best
            self.adjust_largest_object_regularity(self.props, self.labels, noise_suppression_var, best)

    def adjust_largest_object_regularity(self, props, labels, noise_suppression_var, canditate) -> RegionProperties:
        props = regionprops(labels)
        p = next(p for p in props if p.label == canditate)

        print(f"Solidity          : {p.solidity:.2f}")
        print(f"Euler             : {p.euler_number:.2f}  \n -----------")
        #print(f"Circularity       : {4*np.pi*p.area / (p.perimeter**2):.2f} ")

        previous_euler = p.euler_number

        while p.euler_number != 0.00 and self.check_if_better(previous_euler, p.euler_number) and noise_suppression_var > 0.005:
            previous_euler = p.euler_number
            noise_suppression_var -= 0.005
            props, labels = self.make_binary(noise_suppression_var)
            largest = max(props, key=lambda p: p.area).label
            props = regionprops(labels)
            p = next(p for p in props if p.label == largest)

            print(f"Solidity          : {p.solidity:.2f}")
            print(f"Euler             : {p.euler_number:.2f}  ")
            #print(f"Circularity       : {4*np.pi*p.area / (p.perimeter**2):.2f} ")

        self.props = props
        self.labels = labels

        return p


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
        # Threshold to binary
        self.binary_global = self.gray_smooth - noise_suppression_var > self.thresh
        # Remove small objects (noise) and fill small holes
        cleaned = morphology.remove_small_objects(self.binary_global, min_size=500)
        cleaned = morphology.remove_small_holes(cleaned, area_threshold=500)

        # Further clean-up
        selem = morphology.disk(3)
        cleaned = morphology.binary_closing(cleaned, selem)

        labels, num = ndi.label(cleaned)
        props = regionprops(labels)
        if len(props) == 0 and self.counter < 5:
            print("fu")
            self.counter += 1
            self.make_binary(noise_suppression_var - 0.005)
        if len(props) == 0 and self.counter >= 5:
            print("No object was detected")
            return None
        return props, labels

    def plot_results(self):
        fig, axes = plt.subplots(1, 3, figsize=(16,4))
        for ax in axes: ax.axis('off')

        axes[0].imshow(self.gray, cmap='gray')
        axes[0].set_title('Grayscale input')
        axes[1].imshow(self.binary_global, cmap='gray')
        axes[1].set_title('Raw binary')

        #axes[2].imshow(cleaned, cmap='gray')
        #axes[2].set_title('All objects')

        #largest = max(self.props, key=lambda p: p.area).label
        biggest_mask = (self.labels == self.candidate)

        axes[2].imshow(biggest_mask, cmap='gray')
        axes[2].set_title('Cell only')

        plt.tight_layout()
        #plt.show()
        return fig

