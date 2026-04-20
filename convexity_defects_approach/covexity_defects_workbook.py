import marimo

__generated_with = "0.21.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    from convexity_defects import get_and_split_all_labels, plot_cells_w_numbers, plot_gradient_heatmap_with_lines, plot_line_gradient_profile, analyze_line_gradients_global_min

    return (
        analyze_line_gradients_global_min,
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
    depth = mo.ui.slider(10, 110, value=50, label="Depth threshold")

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
    final_labels, deep_defects, all_line_gradients, grad_mag_smooth, markers, all_hulls = get_and_split_all_labels( "./YPGal/Snap-8145.czi", min_dist.value, sigma_grad.value, depth.value, k=2.0)
    return (
        all_hulls,
        all_line_gradients,
        deep_defects,
        final_labels,
        grad_mag_smooth,
        markers,
    )


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
def _(all_hulls, all_line_gradients, grad_mag_smooth, markers, mo):
    import matplotlib.pyplot as plt
    import numpy as np
    import matplotlib.pyplot as plt
    import numpy as np

    def plot_gradient_heatmap_with_peaks(
        grad_mag_smooth,
        markers,
        line_gradients,
        all_hulls=None,
        peak_marker_size=80,
        peak_color="lime",
        hull_color="deepskyblue",
        hull_linewidth=2,
    ):
        """
        Plot gradient heatmap with watershed peaks, cut lines, and optional convex hulls,
        without matplotlib white frame.

        Parameters
        ----------
        grad_mag_smooth : np.ndarray
            2D gradient magnitude image.
        markers : np.ndarray | list | tuple
            Either:
            - 2D marker image (same size as image), nonzero values = marker locations
            - list/array of (x, y) coordinates
        line_gradients : list[dict]
            Each item should contain "pt1" and "pt2".
        all_hulls : list[dict] | list[np.ndarray] | None
            Optional hulls to plot.
            Supported formats:
            - [{"original_label": ..., "hull_points": ndarray(N,2)}, ...]
            - [ndarray(N,2), ndarray(M,2), ...]
        """
        fig, ax = plt.subplots(figsize=(6, 6), frameon=False)
        ax.imshow(grad_mag_smooth, cmap="inferno")

        peak_x = []
        peak_y = []

        if markers is not None:
            markers = np.asarray(markers)

            if markers.ndim == 2:
                # marker image
                rows, cols = np.where(markers > 0)
                peak_x = cols
                peak_y = rows

            elif markers.ndim == 2 and markers.shape[1] == 2:
                # coordinate array [(x, y), ...]
                peak_x = markers[:, 0]
                peak_y = markers[:, 1]

            elif isinstance(markers, (list, tuple)) and len(markers) > 0:
                peak_x = [p[0] for p in markers]
                peak_y = [p[1] for p in markers]

        if len(peak_x) > 0:
            ax.scatter(
                peak_x,
                peak_y,
                s=peak_marker_size,
                c=peak_color,
                marker="o",
                linewidths=0.5,
                zorder=5,
            )

        # Plot cut lines
        if line_gradients is not None:
            for item in line_gradients:
                pt1 = item["pt1"]
                pt2 = item["pt2"]

                x1, y1 = pt1
                x2, y2 = pt2

                ax.plot([x1, x2], [y1, y2], color="cyan", linewidth=3, zorder=6)
                ax.plot(x1, y1, "wo", markersize=4, zorder=7)
                ax.plot(x2, y2, "wo", markersize=4, zorder=7)

        # Plot convex hulls
        if all_hulls is not None:
            for hull_item in all_hulls:
                if isinstance(hull_item, dict):
                    hull = hull_item.get("hull_points", None)
                else:
                    hull = hull_item

                if hull is None:
                    continue

                hull = np.asarray(hull)
                if hull.ndim != 2 or hull.shape[1] != 2 or len(hull) < 2:
                    continue

                x = hull[:, 0]
                y = hull[:, 1]

                # close the polygon
                x_closed = np.append(x, x[0])
                y_closed = np.append(y, y[0])

                ax.plot(
                    x_closed,
                    y_closed,
                    color=hull_color,
                    linewidth=hull_linewidth,
                    zorder=4,
                )

        ax.set_axis_off()
        for spine in ax.spines.values():
            spine.set_visible(False)

        ax.set_xticks([])
        ax.set_yticks([])

        fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
        ax.set_position([0, 0, 1, 1])
        fig.patch.set_alpha(0)
        ax.patch.set_alpha(0)

        return fig

    mo.mpl.interactive(plot_gradient_heatmap_with_peaks(grad_mag_smooth, markers, all_line_gradients, all_hulls))
    return


@app.cell
def _(all_line_gradients, mo, plot_line_gradient_profile):
    mo.mpl.interactive(plot_line_gradient_profile(all_line_gradients))
    return


@app.cell
def _(all_line_gradients, analyze_line_gradients_global_min):
    results = analyze_line_gradients_global_min(all_line_gradients)

    for r in results:
        print(f"Line {r['line_index']} {r['pt1']} -> {r['pt2']}")
        if "error" in r:
            print(" ", r["error"])
            continue

        print(f"  First max from left: idx={r['max_idx']}, value={r['max_val']:.4f}")
        print(f"  Global minimum: idx={r['global_min_idx']}, value={r['global_min_val']:.4f}")
        print(
            f"  Highest increase between them: {r['largest_increase_derivative']:.4f} "
            f"(from idx {r['largest_increase_from_idx']} to {r['largest_increase_to_idx']})"
        )
        print(f"  Total change max->global min: {r['total_change_max_to_global_min']:.4f}")
        print(f"  Total drop max->global min: {r['total_drop_max_to_global_min']:.4f}")
    return


@app.cell
def _(all_line_gradients):
    print(min(all_line_gradients[0]["values"]))
    print(max(all_line_gradients[0]["values"]))
    return


if __name__ == "__main__":
    app.run()
