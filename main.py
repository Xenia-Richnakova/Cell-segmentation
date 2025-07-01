from process import Cell_segmentation, ImgFormat
import time

start = time.time()
##### Prepared Use cases #####

# Segment multiple pictures in CZI format
# more little cells on one picture, lower sigma value recomended
#c = Cell_segmentation(False, "cell_pictures/zeiss_pics", ImgFormat.CZI, "czi_segmentations.pdf", 0.4)
# -----------------------------------------------------------


# Segment multiple pictures in TIF format
# less bigger cells on one picture, higher sigma value recomended
#c = Cell_segmentation(False, "cell_pictures/secondTry/pics", ImgFormat.TIF, "tif_combined.pdf", k=0.3)
# -----------------------------------------------------------

end = time.time()
elapsed = end - start

print(f'{int(elapsed // 60)} minutes {elapsed % 60:.2f} seconds')

# Segment and plot single picture in TIF format
p = "cell_pictures/secondTry/pics/light_pick18.tif"
#c = Cell_segmentation(True, p, ImgFormat.TIF, "", k=0.25)
# -----------------------------------------------------------


# Segment and plot single picture in CZI format
k = "zeiss_pics/Snap-6293.czi"
#c = Cell_segmentation(True, k, ImgFormat.CZI, "")

k = "cell_pictures/zeiss_pics/Snap-6268.czi"
#c = Cell_segmentation(True, k, ImgFormat.CZI, "")


# Folder with original pics into folder with segmented pngs
cs = Cell_segmentation(
    single_image=False,
    path="cell_pictures/zeiss_pics",
    img_format=ImgFormat.CZI,
    name_of_final_doc="cells",
    k=0.3
)
cs.create_pngs("segmented_pngs", to_fill=False)
