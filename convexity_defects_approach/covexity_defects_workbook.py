import marimo

__generated_with = "0.18.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    from convexity_defects import get_and_split_all_labels, plot_cells_w_numbers
    return get_and_split_all_labels, mo, plot_cells_w_numbers


@app.cell
def _(mo):
    sigma_grad = mo.ui.slider(0.1, 100, 0.1, 22, label="Sigma (gradient smoothing)")
    min_dist = mo.ui.slider(1, 300, value=220, label="Min distance")
    depth = mo.ui.slider(1, 30, value=20, label="Min distance")

    ui = mo.vstack([sigma_grad, min_dist, depth])
    return depth, min_dist, sigma_grad, ui


@app.cell
def _(ui):
    ui
    return


@app.cell
def _(depth, get_and_split_all_labels, min_dist, sigma_grad):
    #  "../YPGal/Snap-8149.czi"
    # "../YPGal/Snap-8145.czi"
    final_labels, deep_defects = get_and_split_all_labels( "../YPGal/Snap-8149.czi", min_dist.value, sigma_grad.value, depth.value)
    return deep_defects, final_labels


@app.cell
def _(deep_defects, final_labels, mo, plot_cells_w_numbers):
    mo.mpl.interactive(plot_cells_w_numbers(final_labels, deep_defects))
    return


if __name__ == "__main__":
    app.run()
