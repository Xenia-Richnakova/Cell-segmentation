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
    import matplotlib as plt

    from segment_with_gradient_magnitude import Segment_With_Magnitude
    return EdgeFinder, mo, np, regionprops


@app.cell
def _(EdgeFinder):
    path = "./YPGal/Snap-8145.czi"
    pic = EdgeFinder(path, k=0.45)
    return (pic,)


@app.cell
def _():
    #seg.fill()
    return


@app.cell
def _(mo):

    sigma_grad = mo.ui.slider(0.1, 100, 0.1,22, label="Sigma (gradient smoothing)")
    min_dist = mo.ui.slider(1, 300, value=120, label="Min distance")

    ui = mo.vstack([sigma_grad, min_dist])
    return min_dist, sigma_grad


@app.cell
def _(min_dist, sigma_grad):
    sigma_grad, min_dist
    return


@app.cell
def _(min_dist, regionprops, sigma_grad):
    def isBadSolidity(big_regions):
        for cell in big_regions:
            if cell.solidity < 0.9:
                return True
        return False


    def adjust_watershed(pic):
        min_d = min_dist.value - 20
        end = min_d + 100
        sigma_g = sigma_grad.value
        while True:
            min_d += 20
            sigma_g += 2
            labels, markers = pic.watershed(
                min_distance=min_d,
                sigma_grad=sigma_grad.value,
            )
            props = regionprops(labels)
            big_regions = pic.consider_largest_regions(props)
            print(f"current Min distance: {min_d}, len(bigRegions): {len(big_regions)}, len(pic.coords)): {len(pic.coords)}")

            if (len(big_regions) == len(pic.coords) or min_d > end):
                break



        while (isBadSolidity(big_regions) and min_d < end):
            min_d += 20
            sigma_g += 2
            labels, markers = pic.watershed(
                min_distance=min_d,
                sigma_grad=sigma_grad.value,
            )
            props = regionprops(labels)
            big_regions = pic.consider_largest_regions(props)
            print(f"current Min distance: {min_d}, len(bigRegions): {len(big_regions)}, len(pic.coords)): {len(pic.coords)}")

        return labels, big_regions
    return


@app.cell
def _(min_dist, mo, pic, regionprops, sigma_grad):
    #labels, big_regions = adjust_watershed(pic)
    labels, markers = pic.watershed(min_dist.value, sigma_grad=sigma_grad.value)
    big_regions = regionprops(labels)
    # = pic.consider_largest_regions(props)
    mo.mpl.interactive(pic.plot_cells_w_numbers(labels, big_regions))
    return big_regions, labels


@app.cell
def _(np):

    from skimage.measure import find_contours, EllipseModel

    def contour_features(region_mask, angle_thresh_deg=10):
        contours = find_contours(region_mask.astype(float), 0.5)
        if not contours:
            return None

        contour = max(contours, key=len)   # (N, 2), columns are y, x
        y = contour[:, 0]
        x = contour[:, 1]
        points = np.column_stack((x, y))

        # --- ellipse fit  ---
        ell = EllipseModel.from_estimate(points)
        if ell is None:
            return None

        residuals = ell.residuals(points)
        mean_residual = float(np.mean(residuals))

        # --- straight-edge detection ---
        # tangent direction between consecutive contour points
        dx = np.diff(x, append=x[0])
        dy = np.diff(y, append=y[0])
        seg_angle = np.arctan2(dy, dx)

        # change in tangent direction
        dtheta = np.diff(seg_angle, append=seg_angle[0])
        dtheta = (dtheta + np.pi) % (2 * np.pi) - np.pi  # wrap to [-pi, pi]
        dtheta_deg = np.abs(np.degrees(dtheta))

        # nearly straight where tangent changes only a little
        straight = dtheta_deg < angle_thresh_deg

        # longest consecutive run of straight samples
        # duplicate once to handle wrap-around
        straight2 = np.concatenate([straight, straight])
        longest = 0
        cur = 0
        for v in straight2:
            if v:
                cur += 1
                longest = max(longest, cur)
            else:
                cur = 0
        longest = min(longest, len(straight))

        straight_fraction = longest / len(straight)

        return {
            "mean_residual": mean_residual,
            "straight_fraction": straight_fraction,
            "center": ell.center,
            "axis_lengths": ell.axis_lengths,
            "theta": ell.theta,
        }
    return (contour_features,)


@app.cell
def _(big_regions, contour_features, labels):
    def show_elliptic_features(big_regions, labels):
        sum = 0
        for r in big_regions:
            mask = (labels == r.label)
            f = contour_features(mask)

            if f is None:
                print(f"Region {r.label}: no fit")
                continue

            print(
                f"Region {r.label}: "
                f"residual={f['mean_residual']:.3f}, "
                f"straight_fraction={f['straight_fraction']:.3f}"
            )
            sum += f['straight_fraction']

        print(f"avg: {sum /len(big_regions)}")


    def get_touching_pairs(label_img):
        pairs = set()

        # horizontal neighbors
        a = label_img[:, :-1]
        b = label_img[:, 1:]
        mask = (a != b) & (a > 0) & (b > 0)
        for x, y in zip(a[mask], b[mask]):
            pairs.add(tuple(sorted((int(x), int(y)))))

        # vertical neighbors
        a = label_img[:-1, :]
        b = label_img[1:, :]
        mask = (a != b) & (a > 0) & (b > 0)
        for x, y in zip(a[mask], b[mask]):
            pairs.add(tuple(sorted((int(x), int(y)))))

        return sorted(pairs)


    print(get_touching_pairs(labels))

    def merge_two_touching_objects(label_img, label1, label2):

        # merge label2 into label1
        new_label_img = label_img.copy()
        new_label_img[new_label_img == label2] = label1

        return new_label_img

    show_elliptic_features(big_regions, labels)


    def decide_merge(labels, big_regions):
        labels_of_touching = get_touching_pairs(labels)

        for pair in labels_of_touching:
            label_0 = pair[0]
            label_1 = pair[1]

            if label_1 == 12:
                print(
                        f"LABEL 11,12"
                        f"Merged Regions {label_0} and {label_1}: "
                    )

            # Generate masks for both
            mask_for0 = (labels == label_0)
            f0 = contour_features(mask_for0)

            mask_for1 = (labels == label_1)
            f1 = contour_features(mask_for1)

            if f0 is None:
                label_0 -= 1
                mask_for0 = (labels == label_0)
                f0 = contour_features(mask_for0)
            if f0 is None or f1 is None:
                continue

            # Generate mask and features for the merged pair
            new_labels = merge_two_touching_objects(labels, label_0, label_1)
            mask_merged = (new_labels == label_0)
            f_merged = contour_features(mask_merged)

            if f_merged is None:
                continue

            if label_1 == 12:
                print(
                        f"LABEL 11,12"
                        f"Merged Regions {label_0} and {label_1}: "
                        f"residual={f_merged['mean_residual']:.3f}, "
                        f"straight_fraction={f_merged['straight_fraction']:.3f}"
                    )

            # Decision logic
            if f_merged['straight_fraction'] <= min(f0['straight_fraction'], f1['straight_fraction']):
                if f0['mean_residual'] + f1['mean_residual'] > f_merged['mean_residual']:
                    # Update the main labels array
                    labels = new_labels
                    print(
                        f"Merged Regions {label_0} and {label_1}: "
                        f"residual={f_merged['mean_residual']:.3f}, "
                        f"straight_fraction={f_merged['straight_fraction']:.3f}"
                    )

        return labels




    new_labels = decide_merge(labels, big_regions)
    #llnew_labels = merge_two_touching_objects(labels, 2, 1)
    #new_labels = merge_two_touching_objects(llnew_labels, 4, 5)
    #show_elliptic_features(big_regions, new_labels)


    return (new_labels,)


@app.cell
def _(big_regions, mo, new_labels, pic):
    #mo.mpl.interactive(pic.plot_cells_w_numbers(labels, big_regions))
    mo.mpl.interactive(pic.plot_cells_w_numbers(new_labels, big_regions))
    return


@app.cell
def _(big_regions, np):
    import math

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

        nu = big_regions[i].moments_normalized

        # The 3rd-order moments (3,0 and 0,3) measure asymmetry/skewness
        asymmetry_score = abs(nu[3, 0]) + abs(nu[0, 3])

        print(f"Asymmetry Score: {asymmetry_score}")
        #ellipse_area = math.pi * big_regions[i].axis_minor_length * big_regions[i].axis_major_length
        #print(f"    - {big_regions[i].area / ellipse_area}")


    print("###################################### Cell 1 ")

    for attr in dir(big_regions[0]):
        if not attr.startswith("_"):  # skip private attributes
            try:
                value = getattr(big_regions[0], attr)
                if not callable(value):   # skip methods
                    print(f"{attr}: {value}")
            except:
                continue
    print("###################################### Cell 2 ")
    for attr in dir(big_regions[1]):
        if not attr.startswith("_"):  
            try:
                value = getattr(big_regions[1], attr)
                if not callable(value):   # skip methods
                    print(f"{attr}: {value}")
            except:
                continue

    print("###################################### Cell 3 ")
    for attr in dir(big_regions[2]):
        if not attr.startswith("_"):  
            try:
                value = getattr(big_regions[2], attr)

                if not callable(value):   # skip methods
                    print(f"{attr}: {value}")
            except:
                continue
    return


@app.cell
def _(mo, pic, sigma_grad):

    mo.mpl.interactive(pic.show_heat_map_gradient(sigma_grad.value))
    return


@app.cell
def _(mo, pic):
    mo.mpl.interactive(pic.show_heat_map(1))
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
