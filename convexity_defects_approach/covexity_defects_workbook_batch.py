import marimo

__generated_with = "0.18.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    from pathlib import Path
    from convexity_defects import get_and_split_all_labels, plot_cells_w_numbers
    return Path, get_and_split_all_labels, mo, np


@app.cell
def _(mo):
    folder = mo.ui.text(value="../little_folder", label="Folder with images")
    sigma_grad = mo.ui.slider(0.1, 100, 0.1, 22, label="Sigma (gradient smoothing)")
    min_dist = mo.ui.slider(1, 300, value=220, label="Min distance")
    depth = mo.ui.slider(1, 30, value=20, label="Depth threshold")
    brightness_threshold = mo.ui.slider(100, 2000, value=1500, label="Brightness threshold")

    ui = mo.vstack([folder, sigma_grad, min_dist, depth, brightness_threshold])
    return brightness_threshold, depth, folder, min_dist, sigma_grad, ui


@app.cell
def _(ui):
    ui
    return


@app.cell
def _(np):
    from skimage.io import imread

    try:
        import czifile
    except ImportError:
        czifile = None

    def load_image_for_brightness(path):
        path = str(path)
        lower = path.lower()

        if lower.endswith(".czi"):
            if czifile is None:
                raise ImportError(
                    "For .czi brightness reading, install czifile: pip install czifile"
                )
            arr = czifile.imread(path)
        else:
            arr = imread(path)

        arr = np.asarray(arr)

        # remove singleton dimensions
        arr = np.squeeze(arr)

        # if multi-channel, average over channels
        if arr.ndim >= 3:
            arr = arr.mean(axis=tuple(range(arr.ndim - 2)))

        return arr.astype(np.float32)

    def mean_brightness(path):
        img = load_image_for_brightness(path)
        return float(img.mean())
    return (mean_brightness,)


@app.cell
def _(
    Path,
    brightness_threshold,
    depth,
    folder,
    get_and_split_all_labels,
    mean_brightness,
    min_dist,
    mo,
    sigma_grad,
):
    valid_suffixes = {".czi", ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}

    folder_path = Path(folder.value)
    if not folder_path.exists():
        mo.md(f"Folder does not exist: `{folder.value}`")
        results = {}
        processed_rows = []
    else:
        files = sorted(
            p for p in folder_path.iterdir()
            if p.is_file() and p.suffix.lower() in valid_suffixes
        )

        results = {}
        processed_rows = []

        for img_path in files:
            try:
                brightness = mean_brightness(img_path)
            except Exception as e:
                processed_rows.append(
                    {
                        "file": img_path.name,
                        "brightness": None,
                        "status": f"brightness failed: {e}",
                    }
                )
                continue

            if brightness <= brightness_threshold.value:
                processed_rows.append(
                    {
                        "file": img_path.name,
                        "brightness": round(brightness, 2),
                        "status": "skipped (too dark)",
                    }
                )
                continue

            try:
                final_labels, deep_defects = get_and_split_all_labels(
                    str(img_path),
                    min_dist.value,
                    sigma_grad.value,
                    depth.value,
                )
                results[img_path.name] = (final_labels, deep_defects)
                processed_rows.append(
                    {
                        "file": img_path.name,
                        "brightness": round(brightness, 2),
                        "status": f"processed, cells={int(final_labels.max())}",
                    }
                )
            except Exception as e:
                processed_rows.append(
                    {
                        "file": img_path.name,
                        "brightness": round(brightness, 2),
                        "status": f"segmentation failed: {e}",
                    }
                )

    table = mo.ui.table(processed_rows)
    image_names = list(results.keys())
    selected = mo.ui.dropdown(
        options=image_names,
        value=image_names[0] if image_names else None,
        label="Processed image to display",
    )
    return selected, table


@app.cell
def _(selected, table):
    table
    selected
    return


app._unparsable_cell(
    r"""
    if selected.value is None:
        mo.md(\"No image passed the brightness threshold.\")
        return

    final_labels, deep_defects = results[selected.value]
    mo.mpl.interactive(plot_cells_w_numbers(final_labels, deep_defects))
    """,
    name="_"
)


if __name__ == "__main__":
    app.run()
