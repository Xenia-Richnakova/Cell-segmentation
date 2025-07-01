import numpy as np
from scipy.ndimage import binary_fill_holes
from skimage.measure import regionprops
from skimage.segmentation import find_boundaries

from object_extractor import objectExtractor, select_the_most_regular
from pathlib import Path
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import matplotlib.pyplot as plt
from czifile import CziFile
from enum import Enum
import regex



class ImgFormat(Enum):
    PNG = "png"
    TIF = "tif"
    CZI = "czi"

class Cell_segmentation:
    def __init__(self, single_image : bool, path : str, img_format : ImgFormat, name_of_final_doc : str, sigma_value=1.6, k=0.2) -> None:
        self.single_image = single_image
        self.k = k
        self.path = path
        self.name_of_final_doc = name_of_final_doc
        self.sigma_value = sigma_value

        # handle format
        if img_format is ImgFormat.PNG:
            self.img_format = 'png'
        elif img_format is ImgFormat.TIF:
            self.img_format = 'tif'
        elif img_format is ImgFormat.CZI:
            self.img_format = 'czi'
        else:
            raise ValueError(f"Unsupported format: {img_format}")

        # if you want to plot only one image
        if self.single_image:
            match = regex.findall(r'[^\/]*\.', self.path)
            file_name = match[0][:-1]
            # if czi
            if self.img_format == 'czi':
                czi  = CziFile(self.path)
                data = czi.asarray()
                arr2d = data[0, :, :, 0]

                ext = objectExtractor(image_array=arr2d, sigma_value=self.sigma_value, k=self.k)
                ext.plot_results(file_name, n_best=10, do_plot=self.single_image)

            # if png or tif
            else:
                pic = objectExtractor(self.path, sigma_value=self.sigma_value, k=self.k)
                pic.plot_results(file_name, n_best=10, do_plot=self.single_image)
        # if images should be saved into pdf
        else:
            if self.img_format == "czi":
                self.load_czi()
            else:
                self.load_tif_png()


    def flush_vertical(self, figs, pdf, page_size=(8.27, 11.69)):
        """
        Always make a 3×1 portrait page.
        Plot each fig into the top N slots; delete the extra axes
        so the remainder of the page is just empty white.
        """
        # 1) Create a 3×1 grid
        comp_fig, axes = plt.subplots(3, 1, figsize=page_size,
                                    constrained_layout=True)
        axes = axes.flatten()

        # 2) Fill the first len(figs) slots
        for ax, f in zip(axes, figs):
            canvas = FigureCanvas(f)
            canvas.draw()
            width, height = canvas.get_width_height()
            buf = canvas.buffer_rgba()
            img = (np.frombuffer(buf, dtype="uint8")
                    .reshape(height, width, 4)[..., :3])
            ax.imshow(img)
            ax.axis("off")
            plt.close(f)

        # 3) Remove the unused axes entirely
        for ax in axes[len(figs):]:
            comp_fig.delaxes(ax)

        # 4) Save and close
        pdf.savefig(comp_fig, dpi=600, bbox_inches="tight")
        plt.close(comp_fig)

    def create_pngs(self, output_folder: str, n_best: int = 10, to_fill=False):
        """
        Segment each image in self.path and save a single‐panel PNG
        showing the filled best regions for that image.
        """
        outdir = Path(output_folder)
        outdir.mkdir(parents=True, exist_ok=True)

        folder  = Path(self.path)
        pattern = "*.czi" if self.img_format == 'czi' else f"*.{self.img_format}"
        files   = sorted(folder.glob(pattern))

        for idx, file in enumerate(files, start=1):
            print(f"[{idx}/{len(files)}] {file.name}")
            # 1) load the image & segment
            if self.img_format == 'czi':
                czi   = CziFile(str(file))
                data  = czi.asarray()
                arr2d = data[0, :, :, 0]
                ext   = objectExtractor(image_array=arr2d,
                                        sigma_value=self.sigma_value, k=self.k)
            else:
                ext = objectExtractor(image_path=file,
                                      sigma_value=self.sigma_value, k=self.k)

            try:
                labels_copy = ext.labels.copy()
                props       = regionprops(ext.labels)
                best_labels = []
                for _ in range(n_best):
                    lbl = select_the_most_regular(props, labels_copy)
                    if lbl is None:
                        break
                    best_labels.append(lbl)
                    labels_copy[labels_copy == lbl] = 0

                # 2) build one combined binary mask
                combined_mask = np.isin(ext.labels, best_labels)

                fig, ax = plt.subplots(figsize=(6, 6))
                # Contours Only
                if to_fill:
                    # Filled mask
                    filled        = binary_fill_holes(combined_mask)
                    # find pixel‐wide boundaries between labels
                    bnd = find_boundaries(combined_mask, mode='inner')
                    ax.axis('off')
                    ax.imshow(filled, cmap='gray', vmin=0, vmax=1)
                    ax.set_facecolor('black')

                    ax.contour(
                    bnd,
                    levels=[0.5],
                    colors='red',
                    linewidths=1
                    )
                else:
                    ax.axis('off')
                    ax.imshow(combined_mask, cmap='gray')

                plt.tight_layout()
                # save
                outdir = Path(output_folder)
                outdir.mkdir(exist_ok=True, parents=True)
                out_png = outdir / f"{file.stem}.png"
                fig.savefig(str(out_png), dpi=300, bbox_inches="tight")
                plt.close(fig)


            except Exception as e:
                print(f"  – segmentation failed, skipping ({e})")

        print(f"All segmented‐cell PNGs saved to: {outdir.resolve()}")


    def load_czi(self):
        folder = Path(f'{self.path}')
        czis   = sorted(folder.glob("*.czi"))
        with PdfPages(self.name_of_final_doc) as pdf:
            figs = []
            for idx, file in enumerate(czis, 1):
                print(f"[{idx}/{len(czis)}] {file.name}")
                czi  = CziFile(str(file))
                data = czi.asarray()
                arr2d = data[0, :, :, 0]
                try:
                    ext = objectExtractor(image_array=arr2d, sigma_value=self.sigma_value, k=self.k)
                    fig = ext.plot_results(str(file.name), 10)
                    figs.append(fig)
                except:
                    print("  – no object detected, skipping.")
                # once we have 3, flush them vertically
                if len(figs) == 3:
                    self.flush_vertical(figs, pdf)
                    figs = []
            # flush any remaining (<3) at the end
            if figs:
                self.flush_vertical(figs, pdf)

        print(f'Provided cells were segmented into {self.name_of_final_doc}')

    def load_tif_png(self):
        folder = Path(f'{self.path}')
        images   = sorted(folder.glob(f'*.{self.img_format}'))

        with PdfPages(self.name_of_final_doc) as pdf:
            figs = []
            for idx, file in enumerate(images, 1):
                print(f"[{idx}/{len(images)}] {file.name}")

                try:
                    ext = objectExtractor(image_path=file, sigma_value=self.sigma_value, k=self.k)
                    fig = ext.plot_results(str(file.name), 10)
                    figs.append(fig)
                except:
                    print("  – no object detected, skipping.")
                # once we have 3, flush them vertically
                if len(figs) == 3:
                    self.flush_vertical(figs, pdf)
                    figs = []

            # flush any remaining (<3) at the end
            if figs:
                self.flush_vertical(figs, pdf)

        print(f'Provided cells were segmented into {self.name_of_final_doc}')





