import marimo

__generated_with = "0.18.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    from convexity_defects import get_and_split_all_labels, plot_cells_w_numbers, plot_gradient_heatmap_with_lines, plot_line_gradient_profile
    return (
        get_and_split_all_labels,
        mo,
        plot_cells_w_numbers,
        plot_gradient_heatmap_with_lines,
        plot_line_gradient_profile,
    )


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
    # ../YPGal/Snap-8103.czi nejde
    # "../YPGal/Snap-8151.czi" nejde
    # "../YPGal/Snap-8153.czi"
    final_labels, deep_defects, all_line_gradients, grad_mag_smooth = get_and_split_all_labels( "../YPGal/Snap-8153.czi", min_dist.value, sigma_grad.value, depth.value)
    return all_line_gradients, deep_defects, final_labels, grad_mag_smooth


@app.cell
def _(deep_defects, final_labels, mo, plot_cells_w_numbers):
    mo.mpl.interactive(plot_cells_w_numbers(final_labels, deep_defects))
    return


@app.cell
def _(
    all_line_gradients,
    grad_mag_smooth,
    mo,
    plot_gradient_heatmap_with_lines,
):
    mo.mpl.interactive(plot_gradient_heatmap_with_lines(grad_mag_smooth, all_line_gradients))
    return


@app.cell
def _(all_line_gradients, mo, plot_line_gradient_profile):
    mo.mpl.interactive(plot_line_gradient_profile(all_line_gradients))
    return


if __name__ == "__main__":
    app.run()
