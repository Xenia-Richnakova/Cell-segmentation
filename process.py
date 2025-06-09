import numpy as np
from object_extractor import objectExtractor
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
    def __init__(self, single_image : bool, path : str, img_format : ImgFormat, name_of_final_doc : str, sigma_value=1.6) -> None:
        self.single_image = single_image
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

                ext = objectExtractor(image_array=arr2d, sigma_value=self.sigma_value)
                ext.plot_results(file_name, n_best=10, do_plot=self.single_image)

            # if png or tif
            else:
                pic = objectExtractor(self.path, sigma_value=self.sigma_value)
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
                    ext = objectExtractor(image_array=arr2d, sigma_value=self.sigma_value)
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
                    ext = objectExtractor(image_path=file, sigma_value=self.sigma_value)
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






