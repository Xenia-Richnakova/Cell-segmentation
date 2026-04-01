import marimo

__generated_with = "0.21.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    from pathlib import Path
    from convexity_defects import get_and_split_all_labels, plot_cells_w_numbers

    return Path, get_and_split_all_labels, mo, np


@app.cell
def _():
    import torch
    print(torch.cuda.is_available())
    print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU")
    return


@app.cell
def _(mo):
    folder = mo.ui.text(value="./little_folder", label="Folder with images")
    sigma_grad = mo.ui.slider(0.1, 100, 0.1, 22, label="Sigma (gradient smoothing)")
    min_dist = mo.ui.slider(1, 300, value=220, label="Min distance")
    depth = mo.ui.slider(1, 100, value=50, label="Depth threshold")
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
                raise ImportError("For .czi files install czifile")
            arr = czifile.imread(path)
        else:
            arr = imread(path)

        arr = np.asarray(arr)
        arr = np.squeeze(arr)

        if arr.ndim >= 3:
            arr = arr.mean(axis=tuple(range(arr.ndim - 2)))

        return arr.astype(np.float32)

    def mean_brightness(path):
        img = load_image_for_brightness(path)
        return float(img.mean())

    return czifile, imread, mean_brightness


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
                final_labels, deep_defects, _, _ = get_and_split_all_labels(
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
    return results, selected, table


@app.cell
def _(selected, table):
    table
    selected
    return


@app.cell
def _(results):
    print(results)
    return


@app.cell
def _(Path, czifile, folder, imread, np, results):
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    pdf_path = "../segmentation_results.pdf"

    with PdfPages(pdf_path) as pdf:

        for image, (labels, defects) in results.items():
            path = str(Path(folder.value) / image)

            # --- load image ---
            if image.lower().endswith(".czi"):
                arr = czifile.imread(path)
            else:
                arr = imread(path)

            arr = np.asarray(arr)
            arr = np.squeeze(arr)

            if arr.ndim >= 3:
                arr = arr.mean(axis=tuple(range(arr.ndim - 2)))

            original = arr.astype(np.float32)

            # --- create figure ---
            fig, axes = plt.subplots(1, 2, figsize=(12, 6))

            axes[0].imshow(original, cmap="gray")
            axes[0].set_title(f"Original: {image}")
            axes[0].axis("off")

            axes[1].imshow(labels, cmap="nipy_spectral")
            axes[1].set_title("Segmentation")
            axes[1].axis("off")

            # --- overlay defects ---
            for defect_idx, d, far_point in defects:
                x, y = far_point
                axes[1].plot(x, y, "wo", markersize=4)
                axes[1].text(x + 5, y - 4, str(defect_idx), color="white", fontsize=8)

            plt.tight_layout()

            # ✅ save this figure as one page in PDF
            pdf.savefig(fig)

            plt.close(fig)  # IMPORTANT to free memory

    print(f"Saved PDF to: {pdf_path}")
    return


@app.cell
def _():
    import os
    print(os.path.abspath("../segmentation_results.pdf"))
    return


if __name__ == "__main__":
    app.run()
