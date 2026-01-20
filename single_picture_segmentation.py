import marimo

__generated_with = "0.18.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    from counting_cells import EdgeFinder
    from skimage.measure import regionprops
    import numpy as np
    import os
    return EdgeFinder, mo, np, regionprops


@app.cell
def _(EdgeFinder):
    path = "./Snap-7254.czi"
    pic = EdgeFinder(path)
    return (pic,)


@app.cell
def _(mo):
    sigma_val = mo.ui.slider(0.1, 10, 0.1, 3, label="Sigma (distance smoothing)")
    sigma_grad = mo.ui.slider(0.1, 10, 0.1, 1.0, label="Sigma (gradient smoothing)")
    min_dist = mo.ui.slider(1, 60, value=20, label="Min distance")

    ui = mo.vstack([sigma_val, sigma_grad, min_dist])
    return min_dist, sigma_grad, sigma_val


@app.cell
def _(min_dist, sigma_grad, sigma_val):
    sigma_val, sigma_grad, min_dist
    return


@app.cell
def _(min_dist, pic, regionprops, sigma_grad, sigma_val):
    labels, markers = pic.watershed(
        min_distance=min_dist.value,
        sigma_dist=sigma_val.value,
        sigma_grad=sigma_grad.value,
    )
    props = regionprops(labels)
    big_regions = pic.consider_largest_regions(props)
    return big_regions, labels


@app.cell
def _(big_regions, labels, mo, pic):
    mo.mpl.interactive(pic.plot_cells_w_numbers(labels, big_regions))
    return


@app.cell
def _(mo):
    out_name = mo.ui.text(value="labels_uint8.png", label="Output PNG filename")
    export_button = mo.ui.button(label="Save to Disk") # Use a button, not a checkbox
    return (out_name,)


@app.cell
def _(labels, mo, np, out_name):
    from PIL import Image
    import io

    def get_png_data():
        # 1. Convert the labels array to an 8-bit image format
        img = Image.fromarray(labels.astype(np.uint8))

        # 2. Save the image to an in-memory buffer
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")

        # 3. Return the raw bytes
        return buffer.getvalue()

    # Create the download component
    download_button = mo.download(
        data=get_png_data,
        filename=out_name.value,
        label=f"Download {out_name.value}",
        disabled=labels is None
    )

    # This is the magic line!
    # Putting the variable name here makes it appear in the UI.
    download_button
    return


if __name__ == "__main__":
    app.run()
