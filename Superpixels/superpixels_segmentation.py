import marimo

__generated_with = "0.18.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    from SLIC_Watershed import segment_cells_slic_watershed
    from read_image import readCZI
    return mo, readCZI, segment_cells_slic_watershed


@app.cell
def _(readCZI, segment_cells_slic_watershed):
    path = "images/Snap-8177.czi"
    image = readCZI(path)

    labels = segment_cells_slic_watershed(path, image)
    return (labels,)


@app.cell
def _(labels, mo):
    import matplotlib.pyplot as plt
    def plot_cells_w_numbers(labels):
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(labels, cmap="nipy_spectral")
        return fig

    mo.mpl.interactive(plot_cells_w_numbers(labels))
    return


if __name__ == "__main__":
    app.run()
