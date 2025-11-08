import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os
def log(msg):
    print(msg, flush=True)
def save_image(img, path):
    Image.fromarray(img).save(path)
def mkdirs(paths):
    for p in paths:
        os.makedirs(p, exist_ok=True)
def create_disk_se(radius):
    y, x = np.ogrid[-radius:radius+1, -radius:radius+1]
    mask = (x**2 + y**2 <= radius**2)
    return mask.astype(np.uint8)
def stretch_contrast(img, low=2, high=98):
    p_low, p_high = np.percentile(img, (low, high))
    img = (img.astype(float) - p_low) / (p_high - p_low) * 255
    return np.clip(img, 0, 255).astype(np.uint8)
def _validate_and_broadcast(img, se):
    se = np.asarray(se)
    if se.dtype != bool:
        se = se != 0
    per_channel = False
    if img.ndim == se.ndim + 1:
        per_channel = True
    elif img.ndim != se.ndim:
        raise ValueError("footprint.ndim must equal img.ndim or img.ndim-1 for channel last images")
    return se, per_channel
def _pad_widths_for_footprint(se_shape):
    center = tuple(s // 2 for s in se_shape)  
    pad = []
    for dim_len, c in zip(se_shape, center):
        pad_before = c
        pad_after = dim_len - c - 1
        pad.append((pad_before, pad_after))
    return tuple(pad), center
def _apply_offsets_min_or_max(padded, img_shape, pad_before, offsets, op='min'):
    views = []
    for off in offsets:
        sl = []
        for ax, o in enumerate(off):
            start = pad_before[ax] + o
            stop = start + img_shape[ax]
            sl.append(slice(start, stop))
        views.append(padded[tuple(sl)])
    stacked = np.stack(views, axis=0)  
    if op == 'min':
        return stacked.min(axis=0)
    else:
        return stacked.max(axis=0)
def _morph_gray(img, se, operation='erosion'):
    img = np.asarray(img)
    se, per_channel = _validate_and_broadcast(img, se)
    if per_channel:
        channels = []
        for c in range(img.shape[-1]):
            channel = _morph_gray(img[..., c], se, operation=operation)
            channels.append(channel)
        return np.stack(channels, axis=-1)
    pad_widths, center = _pad_widths_for_footprint(se.shape)
    padded = np.pad(img, pad_width=pad_widths, mode='reflect')
    true_positions = np.argwhere(se)
    center_arr = np.array(center)
    offsets = [tuple(pos - center_arr) for pos in true_positions]
    if len(offsets) == 0:
        return img.copy()
    pad_before = [p[0] for p in pad_widths]
    if operation == 'erosion':
        res = _apply_offsets_min_or_max(padded, img.shape, pad_before, offsets, op='min')
    else:
        res = _apply_offsets_min_or_max(padded, img.shape, pad_before, offsets, op='max')
    return res.astype(img.dtype, copy=False)
def grayscale_erosion(img, se):
    return _morph_gray(img, se, operation='erosion')
def grayscale_dilation(img, se):
    return _morph_gray(img, se, operation='dilation')
def grayscale_opening(img, se):
    tmp = grayscale_erosion(img, se)
    return grayscale_dilation(tmp, se)
def grayscale_closing(img, se):
    tmp = grayscale_dilation(img, se)
    return grayscale_erosion(tmp, se)
def part_a(image, se_radius=7):
    log("======Part-A processing...======")
    se = create_disk_se(se_radius)
    eros = grayscale_erosion(image, se)
    dil = grayscale_dilation(image, se)
    opening = grayscale_opening(image, se)
    closing = grayscale_closing(image, se)
    outdir = "./Q2_Output/PartA"
    save_image(eros, f"{outdir}/erosion.png")
    save_image(dil, f"{outdir}/dilation.png")
    save_image(opening, f"{outdir}/opening.png")
    save_image(closing, f"{outdir}/closing.png")
    fig, axes = plt.subplots(1, 5, figsize=(15, 4))
    for ax, img_, t in zip(axes, [image, eros, dil, opening, closing], ["Orig", "Erosion", "Dilation", "Opening", "Closing"]):
        ax.imshow(img_, cmap="gray"); ax.set_title(t); ax.axis('off')
    plt.savefig(f"{outdir}/comparison.png"); plt.close()
    return opening, closing, se
def part_b(image, se_radius=15):
    log("======Part-B processing...======")
    se = create_disk_se(se_radius)
    closing = grayscale_closing(image, se)
    bh = closing.astype(np.int32) - image.astype(np.int32)
    bh = np.clip(bh, 0, 255).astype(np.uint8)
    stretched = stretch_contrast(bh, 1, 99)
    outdir = "./Q2_Output/PartB"
    save_image(bh, f"{outdir}/vessels_raw.png")
    save_image(stretched, f"{outdir}/vessels_contrast_stretch.png")
    fig, axes = plt.subplots(1, 3, figsize=(12, 5))
    for ax, img_, t in zip(axes, [image, bh, stretched], ["Orig", "Bottom-Hat", "Contrast"]):
        ax.imshow(img_, cmap="gray"); ax.set_title(t); ax.axis('off')
    plt.savefig(f"{outdir}/comparison.png"); plt.close()
    return stretched
def part_c(vessels_img, se_radius=2):
    log("======Part-C processing...======")
    se = create_disk_se(se_radius)
    gap_filled = grayscale_closing(vessels_img, se)
    gap_filled_stretch = stretch_contrast(gap_filled, 1, 99)
    outdir = "./Q2_Output/PartC"
    save_image(gap_filled, f"{outdir}/vessels_gap_filled.png")
    save_image(gap_filled_stretch, f"{outdir}/gap_filled_stretch.png")
    return gap_filled_stretch
def part_d(vessels_img, max_radius=20):
    log("======Part-D processing...======")
    radii = np.arange(0, max_radius+1)
    sums = []
    for r in radii:
        se = create_disk_se(r)
        opened = grayscale_opening(vessels_img, se)
        sums.append(np.sum(opened))
    sums = np.array(sums)
    pattern_spectrum = -np.diff(sums)
    fig, axes = plt.subplots(2, 1, figsize=(6, 8))
    axes[0].plot(radii, sums)
    axes[0].set_title("Sum after opening vs. SE radius")
    axes[0].set_xlabel("Radius (px)"); axes[0].set_ylabel("Sum")
    axes[1].plot(radii[1:], pattern_spectrum)
    axes[1].set_title("Pattern spectrum (thickness dist.)")
    axes[1].set_xlabel("Radius (px)"); axes[1].set_ylabel("Count");
    plt.tight_layout()
    plt.savefig("./Q2_Output/PartD/granulometry.png")
    plt.close()
    return pattern_spectrum
def main():
    mkdirs(['./Q2_Output', './Q2_Output/PartA', './Q2_Output/PartB', './Q2_Output/PartC', './Q2_Output/PartD'])
    image = np.array(Image.open("../Testcases/q2_input.jpg").convert('L'))
    save_image(image, "./Q2_Output/original_image.png")
    opening, closing, se = part_a(image)
    vessels = part_b(image)
    vessels_filled = part_c(vessels)
    pattern = part_d(vessels_filled)
    log("Processing complete. See ./Q2_Output/")
if __name__ == '__main__':
    main()
