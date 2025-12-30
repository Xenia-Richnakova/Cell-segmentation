import marimo

__generated_with = "0.18.4"
app = marimo.App(
    width="medium",
    layout_file="layouts/edge_segmentation_ntb.grid.json",
)


@app.cell
def _():
    import marimo as mo
    from counting_cells import EdgeFinder
    from skimage.measure import regionprops
    import numpy as np
    return EdgeFinder, mo, np, regionprops


@app.cell
def _(EdgeFinder):

    #path = "/home/xenia-richnakova/Desktop/prifuk/obrazky_PriFUK/Olympus_magMag234.85_2025-02-26/DAPI/1355-8xK_gal_4872.tif" #37
    path = "./moreCells.tif" #78
    #path = "/home/xenia-richnakova/Desktop/prifuk/obrazky_PriFUK/Olympus_magMag234.85_2025-02-26/DAPI/1355-8xK_gal_4878.tif" #20-120

    #path = "/home/xenia-richnakova/Desktop/prifuk/obrazky_PriFUK/Olympus_magMag234.85_2025-02-26/DAPI/1355-8xK_gal_4865.tif"
    #path = "/home/xenia-richnakova/Desktop/prifuk/obrazky_PriFUK/Olympus_magMag234.85_2025-02-26/DAPI/1355-8xK_gal_4874.tif"
    #path = "/home/xenia-richnakova/Desktop/prifuk/obrazky_PriFUK/Olympus_magMag234.85_2025-02-26/DAPI/1355-8xK_gal_4884.tif" #20-120

    pic = EdgeFinder(path)
    return path, pic


@app.cell
def _(mo):
    sigma_val = mo.ui.slider(0.1, 5, 0.1, 3)
    sigma_val
    return (sigma_val,)


@app.cell
def _(mo, sigma_val):
    mo.md(rf"""
    **Sigma value:** {sigma_val.value}
    """)
    return


@app.cell
def _(np, pic, regionprops, s):
    labels, markers = pic.watershed(min_distance=s.value, sigma_dist=5.0)
    props = regionprops(labels)
    areas = np.array([p.area for p in props])
    big_regions = pic.consider_largest_regions(props)
    return big_regions, labels


@app.cell
def _(mo):
    s = mo.ui.slider(1, 120)
    return (s,)


@app.cell
def _(mo, s):
    mo.md(rf"""
    minimal distance: {s.value}

    {s}
    """)
    return


@app.cell
def _(pic):
    pic.watershed()
    return


@app.cell
def _(labels, mo, pic):
    mo.mpl.interactive(pic.plot_ws_overlay(labels))
    return


@app.cell
def _(big_regions, labels, mo, pic):
    mo.mpl.interactive(pic.plot_cells_w_numbers(labels, big_regions))
    return


@app.cell
def _(mo, pic, s):
    {s.value}
    mo.mpl.interactive(pic.show_heat_map())
    return


@app.cell
def _(mo, pic, s):
    {s.value}
    markers_coords = ""
    for i in pic.coords:
        markers_coords += f"{i}\n"
    mo.md(markers_coords)
    return


@app.cell
def _(mo, path, sigma_val):
    from object_extractor import objectExtractor, select_the_most_regular
    from scipy.ndimage import binary_fill_holes
    import matplotlib.pyplot as plt
    from skimage.segmentation import find_boundaries
    oe = objectExtractor(path, sigma_value=sigma_val.value)

    mo.mpl.interactive(oe.plot_results())
    return


if __name__ == "__main__":
    app.run()
