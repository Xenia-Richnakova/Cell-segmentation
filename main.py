from  object_extractor import *
from pathlib import Path
from skimage import io, color, filters, morphology
from scipy import ndimage as ndi
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt

#pic = objectExtractor("pictures/light_pick20.tif")
#pic.plot_results()
folder = Path('pictures')
tifs   = sorted(folder.glob('*.tif'))
counter = 0


with PdfPages('combined_masks.pdf') as pdf:
    for tif in sorted(folder.glob('*.tif')):
        print(counter)
        counter += 1
        extractor = objectExtractor(str(tif))
        fig = extractor.plot_results()        # your original Figure
        pdf.savefig(fig, dpi=600, bbox_inches='tight')
        plt.close(fig)

# 2) Combine them into single PDF
pdf_path = folder / "combined_masks.pdf"
with PdfPages(str(pdf_path)) as pdf:
    for mask_png in sorted(folder.glob('*_mask.png')):
        fig, ax = plt.subplots()
        ax.imshow(io.imread(str(mask_png)), cmap='gray')
        ax.axis('off')
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)



