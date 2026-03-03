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
    path = "./YPD/Snap-7991.czi"
    pic = EdgeFinder(path)
    return (pic,)


@app.cell
def _(mo):
    sigma_val = mo.ui.slider(0.1, 10, 0.1, 0.5, label="Sigma (distance smoothing)")
    sigma_grad = mo.ui.slider(0.1, 30, 0.1,22, label="Sigma (gradient smoothing)")
    min_dist = mo.ui.slider(1, 300, value=140, label="Min distance")

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
def _(big_regions, np):

    def elongation_ratio(p, eps=1e-9):
        l1, l2 = p.inertia_tensor_eigvals
        return l1 / (l2 + eps)

    # empiricke zistenie, vsetko co ma solidity pod 0.9 je nedobre
    for i in range(len(big_regions)):
        ratio = elongation_ratio(big_regions[i])
        score = np.exp(-0.5 * (ratio - 1))
        #for p in big_regions[0]:
        print(f"Cell: {big_regions[i].label} has: ")
        print(f"   - area: {big_regions[i].area}")
        print(f"   - solidity: {big_regions[i].solidity}")
        print(f"   - eccentricity: {big_regions[i].eccentricity}")
        #print(f"   - inertia_tensor: {big_regions[i].inertia_tensor}")
        #print(f"   - inertia_tensor_eigvals: {big_regions[i].inertia_tensor_eigvals}")
        print(f"   - ratio: {ratio}, score: {score}")
        # extent = area / area_bbox
     

    #for p in big_regions[0]:
        #print(p)
    return


@app.cell
def _(mo, pic, sigma_grad):

    mo.mpl.interactive(pic.show_heat_map_gradient(sigma_grad.value))
    return


@app.cell
def _(mo, pic, sigma_val):
    mo.mpl.interactive(pic.show_map_for_distance_transform(sigma_val.value, "vanimo"))
    # MOZNO: Accent

    #'Accent', 'Accent_r', 'Blues', 'Blues_r', 'BrBG', 'BrBG_r', 'BuGn', 'BuGn_r', 'BuPu', 'BuPu_r', 'CMRmap', 'CMRmap_r', 'Dark2', 'Dark2_r', 'GnBu', 'GnBu_r', 'Grays', 'Grays_r', 'Greens', 'Greens_r', 'Greys', 'Greys_r', 'OrRd', 'OrRd_r', 'Oranges', 'Oranges_r', 'PRGn', 'PRGn_r', 'Paired', 'Paired_r', 'Pastel1', 'Pastel1_r', 'Pastel2', 'Pastel2_r', 'PiYG', 'PiYG_r', 'PuBu', 'PuBuGn', 'PuBuGn_r', 'PuBu_r', 'PuOr', 'PuOr_r', 'PuRd', 'PuRd_r', 'Purples', 'Purples_r', 'RdBu', 'RdBu_r', 'RdGy', 'RdGy_r', 'RdPu', 'RdPu_r', 'RdYlBu', 'RdYlBu_r', 'RdYlGn', 'RdYlGn_r', 'Reds', 'Reds_r', 'Set1', 'Set1_r', 'Set2', 'Set2_r', 'Set3', 'Set3_r', 'Spectral', 'Spectral_r', 'Wistia', 'Wistia_r', 'YlGn', 'YlGnBu', 'YlGnBu_r', 'YlGn_r', 'YlOrBr', 'YlOrBr_r', 'YlOrRd', 'YlOrRd_r', 'afmhot', 'afmhot_r', 'autumn', 'autumn_r', 'berlin', 'berlin_r', 'binary', 'binary_r', 'bone', 'bone_r', 'brg', 'brg_r', 'bwr', 'bwr_r', 'cividis', 'cividis_r', 'cool', 'cool_r', 'coolwarm', 'coolwarm_r', 'copper', 'copper_r', 'cubehelix', 'cubehelix_r', 'flag', 'flag_r', 'gist_earth', 'gist_earth_r', 'gist_gray', 'gist_gray_r', 'gist_grey', 'gist_grey_r', 'gist_heat', 'gist_heat_r', 'gist_ncar', 'gist_ncar_r', 'gist_rainbow', 'gist_rainbow_r', 'gist_stern', 'gist_stern_r', 'gist_yarg', 'gist_yarg_r', 'gist_yerg', 'gist_yerg_r', 'gnuplot', 'gnuplot2', 'gnuplot2_r', 'gnuplot_r', 'gray', 'gray_r', 'grey', 'grey_r', 'hot', 'hot_r', 'hsv', 'hsv_r', 'inferno', 'inferno_r', 'jet', 'jet_r', 'magma', 'magma_r', 'managua', 'managua_r', 'nipy_spectral', 'nipy_spectral_r', 'ocean', 'ocean_r', 'pink', 'pink_r', 'plasma', 'plasma_r', 'prism', 'prism_r', 'rainbow', 'rainbow_r', 'seismic', 'seismic_r', 'spring', 'spring_r', 'summer', 'summer_r', 'tab10', 'tab10_r', 'tab20', 'tab20_r', 'tab20b', 'tab20b_r', 'tab20c', 'tab20c_r', 'terrain', 'terrain_r', 'turbo', 'turbo_r', 'twilight', 'twilight_r', 'twilight_shifted', 'twilight_shifted_r', 'vanimo', 'vanimo_r', 'viridis', 'viridis_r', 'winter', 'winter_r'
    return


@app.cell
def _(mo, pic, sigma_val):
    mo.mpl.interactive(pic.show_heat_map(sigma_val.value))
    return


@app.cell
def _(mo):
    out_name = mo.ui.text(value="Snap-8223.png", label="Output PNG filename")
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


@app.cell
def _():
    import subprocess, textwrap
    print(subprocess.check_output(["nvidia-smi"]).decode())
    return


if __name__ == "__main__":
    app.run()
