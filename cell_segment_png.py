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
    import io
    import zipfile
    from PIL import Image
    return EdgeFinder, Image, io, mo, np, os, zipfile


@app.cell
def _(mo):
    folder = mo.ui.text(value="./YPD", label="Folder with .czi files")
    brightness_thr = mo.ui.number(value=1450, label="Average brightness threshold (> )")
    ui = mo.vstack([folder, brightness_thr])
    ui
    return brightness_thr, folder


@app.cell
def _(mo):
    sigma_val = mo.ui.slider(0.1, 10, 0.1, 3, label="Sigma (distance smoothing)")
    sigma_grad = mo.ui.slider(0.1, 100, 0.1, 20, label="Sigma (gradient smoothing)")
    min_dist = mo.ui.slider(1, 300, value=200, label="Min distance")
    mo.vstack([sigma_val, sigma_grad, min_dist])
    return min_dist, sigma_grad, sigma_val


@app.cell
def _(brightness_thr, folder, mo, np, os):
    from czifile import CziFile
    # Collect all .czi files
    folder_path = folder.value.strip()
    if not folder_path:
        folder_path = "."

    if not os.path.isdir(folder_path):
        mo.md(f"❌ Folder not found: `{folder_path}`")
        czi_files = []
    else:
        czi_files = sorted(
            [
                os.path.join(folder_path, f)
                for f in os.listdir(folder_path)
                if f.lower().endswith(".czi")
            ]
        )

    # Compute average brightness per file (fast: just load image and mean)
    stats = []
    for p in czi_files:
        try:
            img = CziFile(p).asarray()
            avg = float(np.mean(img))
            stats.append((p, avg))
        except Exception:
            stats.append((p, None))

    thr = float(brightness_thr.value)
    eligible = [(p, avg) for (p, avg) in stats if (avg is not None and avg > thr)]
    ineligible = [(p, avg) for (p, avg) in stats if (avg is not None and avg <= thr)]
    failed = [(p, avg) for (p, avg) in stats if avg is None]

    # Show summary
    summary_lines = [
        f"Found **{len(czi_files)}** `.czi` files in `{folder_path}`.",
        f"Eligible (avg > {thr}): **{len(eligible)}**",
        f"Ineligible: **{len(ineligible)}**",
        f"Failed to read: **{len(failed)}**",
    ]
    mo.md("\n\n".join(summary_lines))
    return (eligible,)


@app.cell
def _(eligible, mo):
    # Preview which files will be processed
    if not eligible:
        mo.md("No eligible files (nothing to process).")
    else:
        preview = "\n".join([f"- `{p}` (avg={avg:.2f})" for p, avg in eligible[:20]])
        more = "" if len(eligible) <= 20 else f"\n\n…and **{len(eligible)-20}** more."
        mo.md("### Eligible files\n" + preview + more)
    return


@app.cell
def _(mo):
    zip_name = mo.ui.text(value="filtered_pngs_YPD.zip", label="Output ZIP filename")
    zip_name
    return (zip_name,)


@app.cell
def _(
    EdgeFinder,
    Image,
    eligible,
    io,
    min_dist,
    mo,
    np,
    sigma_grad,
    sigma_val,
    zip_name,
    zipfile,
):
    def labels_to_png_bytes(labels: np.ndarray) -> bytes:
        # Ensure uint8 output 
        arr = labels.astype(np.uint8)
        img = Image.fromarray(arr, mode="L")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()

    def build_zip() -> bytes:
        # Create a ZIP in memory containing folder filtered_pngs/<basename>.png
        buf = io.BytesIO()
        out_folder = "filtered_pngs"

        with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for path, _avg in eligible:
                # Run segmentation pipeline
                pic = EdgeFinder(path)

                labels, markers = pic.watershed(
                    min_distance=min_dist.value,
                    sigma_dist=sigma_val.value,
                    sigma_grad=sigma_grad.value,
                )

                # Optional: keep only biggest regions 
                # props = regionprops(labels)
                # big_regions = pic.consider_largest_regions(props)

                png_bytes = labels_to_png_bytes(labels)

                base = path.split("/")[-1]
                stem = base.rsplit(".", 1)[0]
                out_name = f"{out_folder}/{stem}.png"

                zf.writestr(out_name, png_bytes)

        return buf.getvalue()

    download_zip = mo.download(
        data=build_zip,
        filename=zip_name.value,
        label=f"Download ZIP ({zip_name.value})",
        disabled=(len(eligible) == 0),
    )

    download_zip
    return


if __name__ == "__main__":
    app.run()
