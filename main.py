from process import Cell_segmentation, ImgFormat

##### Prepared Use cases #####

# Segment multiple pictures in CZI format
# more little cells on one picture, lower sigma value recomended
#c = Cell_segmentation(False, "Cell-segmentation/zeiss_pics", ImgFormat.CZI, "czi_segmentations_4perpage.pdf", 0.4)
# -----------------------------------------------------------


# Segment multiple pictures in TIF format
# less bigger cells on one picture, higher sigma value recomended
#c = Cell_segmentation(False, "Cell-segmentation/pics", ImgFormat.TIF, "tif_combined.pdf")
# -----------------------------------------------------------

# Segment and plot single picture in TIF format
p = "Cell-segmentation/pics/light_pick22.tif"
c = Cell_segmentation(True, p, ImgFormat.TIF, "")
# -----------------------------------------------------------


# Segment and plot single picture in CZI format
k = "Cell-segmentation/zeiss_pics/Snap-6293.czi"
#c = Cell_segmentation(True, k, ImgFormat.CZI, "")