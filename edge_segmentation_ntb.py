import marimo

__generated_with = "0.18.4"
app = marimo.App(
    width="medium",
    layout_file="layouts/edge_segmentation_ntb.grid.json",
)


@app.cell
def _():
    import marimo as mo
    from counted import EdgeFinder
    from skimage.measure import regionprops
    import numpy as np
    return EdgeFinder, mo, np, regionprops


@app.cell
def _(EdgeFinder):
    path = "./moreCells.tif"
    pic = EdgeFinder(path)
    return (pic,)


@app.cell
def _(np, pic, regionprops, s):
    labels, markers = pic.watershed(min_distance=s.value, sigma_dist=5.0)
    props = regionprops(labels)
    areas = np.array([p.area for p in props])
    big_regions = pic.consider_largest_regions(props)
    return big_regions, labels


@app.cell
def _(mo):
    s = mo.ui.slider(1, 400)
    return (s,)


@app.cell
def _(mo, s):
    mo.md(rf"""
    minimal distance: {s.value}

    {s}
    """)
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


if __name__ == "__main__":
    app.run()
