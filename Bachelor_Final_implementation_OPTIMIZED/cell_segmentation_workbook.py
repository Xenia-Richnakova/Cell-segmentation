import marimo

__generated_with = "0.21.1"
app = marimo.App(width="wide")


@app.cell
def _():
    import io
    import sys
    import tempfile
    import traceback
    import zipfile
    from pathlib import Path

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    from PIL import Image

    # Make local project imports work when this file is placed next to:
    # convexity_defects.py, counting_cells.py, object_extractor.py
    try:
        APP_DIR = Path(__file__).resolve().parent
    except NameError:
        APP_DIR = Path.cwd()

    if str(APP_DIR) not in sys.path:
        sys.path.insert(0, str(APP_DIR))

    from convexity_defects import get_and_split_all_labels

    VALID_IMAGE_SUFFIXES = {".czi", ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
    BRIGHTNESS_THRESHOLD = 1500.0

    TEMP_UPLOAD_DIR = Path(tempfile.gettempdir()) / "marimo_cell_segmentation_uploads"
    TEMP_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    return (
        BRIGHTNESS_THRESHOLD,
        Image,
        Path,
        TEMP_UPLOAD_DIR,
        VALID_IMAGE_SUFFIXES,
        get_and_split_all_labels,
        io,
        mo,
        np,
        plt,
        traceback,
        zipfile,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Cell segmentation UI

    This app has two workflows:

    1. **Single image**: upload one image, tune parameters, preview the segmentation, and download the PNG instance mask.
    2. **Batch processing**: choose multiple images from the folder, tune parameters, choose an output folder, and process after pressing **Process batch**. You can download resulting zip file afterwards.


    Export format:

    - `0` = background
    - `255, 254, 253, ..., 1` = individual cell instances
    """)
    return


@app.cell
def _(Image, Path, TEMP_UPLOAD_DIR, io, np, plt):
    _UPLOAD_PATH_CACHE = {}

    def _safe_filename(name):
        path = Path(name or "uploaded_image")
        suffix = path.suffix or ".png"
        stem = path.stem.replace(" ", "_") or "uploaded_image"
        return stem, suffix


    def save_upload_to_temp(upload_widget, prefix="single"):
        cache_key = prefix

        try:
            value = upload_widget.value
        except Exception:
            value = None

        if not value:
            return _UPLOAD_PATH_CACHE.get(cache_key)

        uploaded = value[0]
        name = getattr(uploaded, "name", None) or "uploaded_image"
        contents = getattr(uploaded, "contents", None)

        if contents is None:
            try:
                contents = upload_widget.contents()
            except Exception:
                contents = None

        if contents is None:
            return _UPLOAD_PATH_CACHE.get(cache_key)

        if isinstance(contents, str):
            contents = contents.encode("utf-8")

        stem, suffix = _safe_filename(name)
        out_path = TEMP_UPLOAD_DIR / f"{prefix}{stem}{suffix}"
        out_path.write_bytes(contents)
        _UPLOAD_PATH_CACHE[cache_key] = out_path
        return out_path


    def save_uploads_to_temp(upload_widget, prefix="batch"):
        cache_key = prefix

        try:
            value = upload_widget.value
        except Exception:
            value = None

        if not value:
            return _UPLOAD_PATH_CACHE.get(cache_key, [])

        saved_paths = []

        for index, uploaded in enumerate(value):
            name = getattr(uploaded, "name", None) or f"uploaded_image_{index + 1}"
            contents = getattr(uploaded, "contents", None)

            if contents is None:
                continue

            if isinstance(contents, str):
                contents = contents.encode("utf-8")

            stem, suffix = _safe_filename(name)
            out_path = TEMP_UPLOAD_DIR / f"{prefix}_{index + 1}_{stem}{suffix}"
            out_path.write_bytes(contents)
            saved_paths.append(out_path)

        _UPLOAD_PATH_CACHE[cache_key] = saved_paths
        return saved_paths


    def labels_to_uint8_instances_reverse(labels):
        """Map labels to 8-bit PNG: 0 background, label 1 -> 255, label 255 -> 1."""
        labels = np.asarray(labels)
        if labels.ndim != 2:
            raise ValueError(f"Expected 2D label mask, got shape {labels.shape}.")

        max_label = int(labels.max()) if labels.size else 0
        if max_label > 255:
            raise ValueError(
                f"This segmentation produced {max_label} instances. "
                "An 8-bit PNG can store at most 255 non-background instances."
            )

        out = np.zeros(labels.shape, dtype=np.uint8)
        mask = labels > 0
        # if bigger contrast needed
        #out[mask] = (256 - labels[mask]*10).astype(np.uint8)
        out[mask] = (256 - labels[mask]).astype(np.uint8)
        return out


    def mask_to_png_bytes(mask_uint8):
        buffer = io.BytesIO()
        Image.fromarray(mask_uint8, mode="L").save(buffer, format="PNG")
        return buffer.getvalue()


    def labels_to_color_png_bytes(labels, cmap_name="nipy_spectral"):
        """Convert the instance-label image to a colorful RGB PNG for download"""
        labels = np.asarray(labels)
        if labels.ndim != 2:
            raise ValueError(f"Expected 2D label mask, got shape {labels.shape}.")

        max_label = int(labels.max()) if labels.size else 0
        normalized = labels.astype(np.float32, copy=False)
        if max_label > 0:
            normalized = normalized / max_label

        rgba = plt.get_cmap(cmap_name)(normalized)
        rgb = (rgba[..., :3] * 255).astype(np.uint8)

        buffer = io.BytesIO()
        Image.fromarray(rgb, mode="RGB").save(buffer, format="PNG")
        return buffer.getvalue()


    def load_display_image(path):
        """Best-effort image loader for preview and brightness checks."""
        path = Path(path)
        lower = path.suffix.lower()

        if lower == ".czi":
            try:
                import czifile
                arr = czifile.imread(str(path))
            except Exception:
                return None
        else:
            try:
                arr = np.asarray(Image.open(path))
            except Exception:
                try:
                    from skimage.io import imread
                    arr = imread(str(path))
                except Exception:
                    return None

        arr = np.asarray(arr)
        arr = np.squeeze(arr)

        while arr.ndim > 2:
            if arr.shape[-1] in (3, 4):
                arr = arr[..., :3].mean(axis=-1)
            else:
                arr = arr.mean(axis=0)

        return arr.astype(np.float32, copy=False)


    def plot_original(image_path):
        original = load_display_image(image_path)
        fig, ax = plt.subplots(figsize=(6, 6))

        if original is None:
            ax.text(
                0.5,
                0.5,
                "Original preview unavailable",
                ha="center",
                va="center",
            )
        else:
            ax.imshow(original, cmap="gray")



        name = image_path.name
        ax.set_title(f"{name}")
        ax.axis("off")
        fig.tight_layout()
        return fig


    def mean_brightness(path):
        arr = load_display_image(path)
        if arr is None:
            raise ValueError("Could not load image for brightness check.")
        return float(np.mean(arr))


    def plot_preview(image_path, labels):
        original = load_display_image(image_path)
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))

        if original is None:
            axes[0].text(
                0.5,
                0.5,
                "Original preview unavailable",
                ha="center",
                va="center",
            )
        else:
            axes[0].imshow(original, cmap="gray")
        axes[0].set_title("Original")
        axes[0].axis("off")

        axes[1].imshow(labels, cmap="nipy_spectral", interpolation="nearest")
        axes[1].set_title(f"Segmentation, instances: {int(labels.max())}")
        axes[1].axis("off")

        fig.tight_layout()
        return fig

    return (
        labels_to_color_png_bytes,
        labels_to_uint8_instances_reverse,
        mask_to_png_bytes,
        mean_brightness,
        plot_original,
        plot_preview,
        save_upload_to_temp,
        save_uploads_to_temp,
    )


@app.cell
def _(
    get_and_split_all_labels,
    labels_to_color_png_bytes,
    labels_to_uint8_instances_reverse,
    mask_to_png_bytes,
    np,
):
    def process_one_image(image_path, min_distance, sigma_gradient, depth_threshold, k=0.3):
        final_labels, deep_defects, all_line_gradients, grad_mag_smooth, markers, all_hulls = (
            get_and_split_all_labels(
                str(image_path),
                int(min_distance),
                float(sigma_gradient),
                float(depth_threshold),
                k=k,
            )
        )

        mask_uint8 = labels_to_uint8_instances_reverse(final_labels)
        png_bytes = mask_to_png_bytes(mask_uint8)
        colorful_png_bytes = labels_to_color_png_bytes(final_labels)

        print("final_labels unique:", np.unique(final_labels))
        mask_uint8 = labels_to_uint8_instances_reverse(final_labels)
        print("mask_uint8 unique:", np.unique(mask_uint8))

        return {
            "labels": final_labels,
            "mask_uint8": mask_uint8,
            "png_bytes": png_bytes,
            "colorful_png_bytes": colorful_png_bytes,
            "cell_count": int(final_labels.max()),
            "deep_defects": deep_defects,
            "all_line_gradients": all_line_gradients,
            "grad_mag_smooth": grad_mag_smooth,
            "markers": markers,
            "all_hulls": all_hulls,
        }

    return (process_one_image,)


@app.cell
def _(mo):
    sigma_grad = mo.ui.slider(
        0.1,
        100.0,
        step=0.1,
        value=22.0,
    )

    min_dist = mo.ui.slider(
        1,
        300,
        step=1,
        value=140,
    )

    depth = mo.ui.slider(
        1,
        150,
        step=1,
        value=50,
    )

    parameter_controls = mo.Html(f"""
    <div style="
        display: grid;
        grid-template-columns: 260px 1fr;
        row-gap: 10px;
        column-gap: 1px;
        align-items: center;
    ">
        <div>Sigma: gradient smoothing</div>
        <div>{sigma_grad}</div>

        <div>Min distance: watershed seeds</div>
        <div>{min_dist}</div>

        <div>Depth threshold: convexity defect</div>
        <div>{depth}</div>
    </div>
    """)
    return depth, min_dist, parameter_controls, sigma_grad


@app.cell(hide_code=True)
def _(VALID_IMAGE_SUFFIXES, mo):
    single_upload = mo.ui.file(
        filetypes=sorted(VALID_IMAGE_SUFFIXES),
        multiple=False,
        kind="area",
        label="Upload one image",
    )

    process_single_button = mo.ui.run_button(
        label="Generate segmented mask",
        kind="success",
    )

    single_panel = mo.vstack(
        [
            mo.md("## 1. Single image"),
            mo.md(
                "Upload an image first. The original image is shown immediately. "
                "Tune the parameters, then click **Generate segmented mask**."
            ),
            mo.hstack([single_upload]),
        ]
    )
    return process_single_button, single_panel, single_upload


@app.cell
def _(single_panel):
    single_panel
    return


@app.cell
def _(mo, plot_original, save_upload_to_temp, single_upload):
    single_image_path = save_upload_to_temp(single_upload, prefix="")

    if single_image_path is None:
        single_original_output = mo.md("Upload an image to display the original preview.").callout(kind="info")
    else:
        single_original_output = mo.vstack(
            [
                #mo.md(f"**Uploaded:** `{single_image_path.name}`").callout(kind="success"),
                mo.mpl.interactive(plot_original(single_image_path)),
            ]
        )

    single_original_output
    return (single_image_path,)


@app.cell
def _(mo, parameter_controls, process_single_button):
    mo.vstack([parameter_controls, process_single_button])
    return


@app.cell
def _(
    depth,
    min_dist,
    mo,
    plot_preview,
    process_one_image,
    process_single_button,
    sigma_grad,
    single_image_path,
    traceback,
):
    if single_image_path is None:
        single_result = None
        single_output = mo.md("Upload an image first, then tune parameters and click **Generate segmented mask**.").callout(kind="info")
    elif not process_single_button.value:
        single_result = None
        single_output = mo.md(
            "Original image is loaded. Tune the parameters, then click **Generate segmented mask**."
        ).callout(kind="info")
    else:
        try:
            single_result = process_one_image(
                single_image_path,
                min_dist.value,
                sigma_grad.value,
                depth.value,
            )
            preview = mo.mpl.interactive(plot_preview(single_image_path, single_result["labels"]))
            download = mo.download(
                data=single_result["png_bytes"],
                filename=f"{single_image_path.stem}_instances.png",
                mimetype="image/png",
                label="Download instance PNG",
            )
            colorful_download = mo.download(
                data=single_result["colorful_png_bytes"],
                filename=f"{single_image_path.stem}_colorful_instances.png",
                mimetype="image/png",
                label="Download colorful instance PNG",
            )
            single_output = mo.vstack(
                [
                    preview,
                    mo.hstack([download, colorful_download]),
                    mo.md(
                        f"**Processed:** `{single_image_path.name}` **Detected instances:** `{single_result['cell_count']}`"
                    ).callout(kind="success")
                ]
            )
            #print(single_output["mask_uint8"])
        except Exception as exc:
            single_result = None
            single_output = mo.md(
                f"**Segmentation failed for `{single_image_path.name}`**\n\n"
                f"```text\n{exc}\n\n{traceback.format_exc()}\n```"
            ).callout(kind="danger")

    single_output
    return


@app.cell
def _(BRIGHTNESS_THRESHOLD, VALID_IMAGE_SUFFIXES, mo, parameter_controls):

    batch_upload = mo.ui.file(
        filetypes=sorted(VALID_IMAGE_SUFFIXES),
        multiple=True,
        kind="area",
        label="Select images from a folder for batch processing",
    )

    batch_folder_text = mo.ui.text(
        value="",
        label="Or type input folder path manually",
        placeholder="./images",
    )

    output_folder = mo.ui.text(
        value="./segmented_output",
        label="Output folder name/path",
    )

    skip_dark_images = mo.ui.checkbox(
        value=True,
        label=f"Skip images with average brightness < {int(BRIGHTNESS_THRESHOLD)}",
    )

    process_batch_button = mo.ui.run_button(
        label="Process batch",
        kind="success",
    )

    batch_panel = mo.vstack(
        [
            mo.md("## 2. Batch processing"),
            mo.md(
                "Select multiple image files from a folder. "
                "Supported suffixes: "
                + ", ".join(f"`{s}`" for s in sorted(VALID_IMAGE_SUFFIXES))
            ),
            batch_upload,
            #batch_folder_text,
            parameter_controls,
            output_folder,
            skip_dark_images,
            process_batch_button,
        ]
    )
    return (
        batch_folder_text,
        batch_panel,
        batch_upload,
        output_folder,
        process_batch_button,
        skip_dark_images,
    )


@app.cell
def _(batch_panel):
    batch_panel
    return


@app.cell
def _(Path, batch_folder_text, batch_upload, save_uploads_to_temp):
    def selected_batch_folder():
        typed = (batch_folder_text.value or "").strip()
        if typed:
            return Path(typed).expanduser()

    batch_input_folder = selected_batch_folder()
    batch_uploaded_files = save_uploads_to_temp(batch_upload, prefix="batch")
    return batch_input_folder, batch_uploaded_files


@app.cell
def _(VALID_IMAGE_SUFFIXES, batch_input_folder, batch_uploaded_files, mo):
    if batch_input_folder is not None:
        if not batch_input_folder.exists():
            batch_files = []
            batch_folder_status = mo.md(f"Input folder does not exist: `{batch_input_folder}`").callout(kind="danger")
        elif not batch_input_folder.is_dir():
            batch_files = []
            batch_folder_status = mo.md(f"Input path is not a folder: `{batch_input_folder}`").callout(kind="danger")
        else:
            batch_files = sorted(
                p for p in batch_input_folder.iterdir()
                if p.is_file() and p.suffix.lower() in VALID_IMAGE_SUFFIXES
            )
            batch_folder_status = mo.md(
                f"Selected folder: `{batch_input_folder}`  \
    "
                f"Found `{len(batch_files)}` image(s)."
            ).callout(kind="success" if batch_files else "warn")
    elif batch_uploaded_files:
        batch_files = sorted(
            p for p in batch_uploaded_files
            if p.is_file() and p.suffix.lower() in VALID_IMAGE_SUFFIXES
        )
        batch_folder_status = mo.md(
            f"Uploaded `{len(batch_files)}` image(s) for batch processing."
        ).callout(kind="success" if batch_files else "warn")
    else:
        batch_files = []
        batch_folder_status = mo.md(
            "Select images from a folder to start batch processing."
        ).callout(kind="info")

    batch_folder_status
    return (batch_files,)


@app.cell
def _(
    BRIGHTNESS_THRESHOLD,
    Path,
    batch_files,
    depth,
    io,
    mean_brightness,
    min_dist,
    mo,
    output_folder,
    process_batch_button,
    process_one_image,
    sigma_grad,
    skip_dark_images,
    traceback,
    zipfile,
):
    import time

    if not process_batch_button.value:
        batch_rows = []
        batch_output = mo.md(
            "Batch processing has not started. Press **Process batch** when you are ready."
        ).callout(kind="info")
    elif not batch_files:
        batch_rows = []
        batch_output = mo.md("No images found to process.").callout(kind="warn")
    else:
        start = time.time()
        print("Start time")

    
        out_dir = Path(output_folder.value).expanduser()
        out_dir.mkdir(parents=True, exist_ok=True)

        batch_rows = []
        saved_paths = []

        for image_path in batch_files:
            try:
                brightness = mean_brightness(image_path)
            except Exception as exc:
                brightness = None
                if skip_dark_images.value:
                    batch_rows.append(
                        {
                            "file": image_path.name,
                            "brightness": "unknown",
                            "instances": None,
                            "output": "",
                            "status": f"skipped, brightness check failed: {exc}",
                        }
                    )
                    continue

            if (
                skip_dark_images.value
                and brightness is not None
                and brightness < BRIGHTNESS_THRESHOLD
            ):
                batch_rows.append(
                    {
                        "file": image_path.name,
                        "brightness": round(brightness, 2),
                        "instances": None,
                        "output": "",
                        "status": f"skipped, brightness < {int(BRIGHTNESS_THRESHOLD)}",
                    }
                )
                continue

            try:
                result = process_one_image(
                    image_path,
                    min_dist.value,
                    sigma_grad.value,
                    depth.value,
                )
                out_path = out_dir / f"{image_path.stem}.png"
                out_path.write_bytes(result["png_bytes"])
                saved_paths.append(out_path)

                batch_rows.append(
                    {
                        "file": image_path.name,
                        "brightness": round(brightness, 2) if brightness is not None else "unknown",
                        "instances": result["cell_count"],
                        "output": str(out_path),
                        "status": "saved",
                    }
                )
            except Exception as exc:
                batch_rows.append(
                    {
                        "file": image_path.name,
                        "brightness": round(brightness, 2) if brightness is not None else "unknown",
                        "instances": None,
                        "output": "",
                        "status": f"failed: {exc}",
                    }
                )
                print(traceback.format_exc())

        saved_count = sum(1 for row in batch_rows if row["status"] == "saved")
        skipped_count = sum(1 for row in batch_rows if str(row["status"]).startswith("skipped"))
        table = mo.ui.table(batch_rows, pagination=True, page_size=10)

        batch_output_items = [
            mo.md(
                f"Batch finished. Saved `{saved_count}` mask(s), skipped `{skipped_count}` image(s), "
                f"total checked `{len(batch_rows)}`. Output folder: `{out_dir}`."
            ).callout(kind="success" if saved_count else "warn"),
        ]

        if saved_paths:
            zip_buffer = io.BytesIO()
            zip_folder_name = out_dir.name or "segmented_output"

            with zipfile.ZipFile(zip_buffer, mode="w", compression=zipfile.ZIP_DEFLATED) as zip_file:
                for saved_path in saved_paths:
                    zip_file.write(saved_path, arcname=f"{zip_folder_name}/{saved_path.name}")

            batch_output_items.append(
                mo.download(
                    data=zip_buffer.getvalue(),
                    filename=f"{zip_folder_name}.zip",
                    mimetype="application/zip",
                    label="Download segmented images ZIP",
                )
            )

        batch_output_items.append(table)
        batch_output = mo.vstack(batch_output_items)
        end = time.time()
        print(f"Elapsed time: {end - start}")


    batch_output
    return


if __name__ == "__main__":
    app.run()
