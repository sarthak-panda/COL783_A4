import os
import math
import argparse
import numpy as np
import cv2
import matplotlib.pyplot as plt

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def save_img(path, img_bgr):
    ensure_dir(os.path.dirname(path) or '.')
    cv2.imwrite(path, img_bgr)

def highlight_boundaries(label_map):
    h, w = label_map.shape
    b = np.zeros((h, w), dtype=bool)
    b[:, :-1] |= (label_map[:, :-1] != label_map[:, 1:])
    b[:-1, :] |= (label_map[:-1, :] != label_map[1:, :])
    return b

def get_manual_seeds(image_bgr, k):
    img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(8, 6))
    plt.imshow(img_rgb)
    plt.title(f"Click {k} points to initialize KMeans centers; close when done")
    plt.axis('off')
    pts = plt.ginput(n=k, timeout=0)
    plt.close()
    if len(pts) < k:
        print(f"Only got {len(pts)} points; expected {k}. Falling back to None.")
        return None
    seeds = [(int(round(x)), int(round(y))) for (x, y) in pts]
    return seeds

def kmeans_rgb_segmentation(image_bgr, k=5, max_iter=100, tol=1.0, init_points=None):
    h, w, _ = image_bgr.shape
    N = h * w
    pixels = image_bgr.reshape((-1, 3)).astype(np.float32)  
    if init_points is not None:
        centers = []
        for (x, y) in init_points:
            if x < 0 or x >= w or y < 0 or y >= h:
                continue
            centers.append(pixels[y * w + x])
        if len(centers) < k:
            idx = np.random.choice(N, k - len(centers), replace=False)
            for i in idx:
                centers.append(pixels[i])
        centers = np.array(centers, dtype=np.float32)[:k]
    else:
        centers = np.zeros((k, 3), dtype=np.float32)
        idx = np.random.choice(N)
        centers[0] = pixels[idx]
        dists = np.sum((pixels - centers[0]) ** 2, axis=1)
        for i in range(1, k):
            probs = dists / dists.sum()
            idx = np.random.choice(N, p=probs)
            centers[i] = pixels[idx]
            newd = np.sum((pixels - centers[i]) ** 2, axis=1)
            dists = np.minimum(dists, newd)
    labels = np.full(N, -1, dtype=np.int32)
    for it in range(max_iter):
        dists = np.sum((pixels[:, None, :] - centers[None, :, :]) ** 2, axis=2)
        new_labels = np.argmin(dists, axis=1)
        changes = np.count_nonzero(new_labels != labels)
        labels = new_labels
        new_centers = np.zeros_like(centers)
        for i in range(k):
            mask = (labels == i)
            cnt = mask.sum()
            if cnt > 0:
                new_centers[i] = pixels[mask].mean(axis=0)
            else:
                new_centers[i] = pixels[np.random.choice(N)]
        center_shift = np.sqrt(np.sum((new_centers - centers) ** 2, axis=1))
        centers = new_centers
        if changes == 0 or center_shift.max() < tol:
            print(f"KMeans converged at iteration {it}; label changes={changes}; max shift={center_shift.max():.4f}")
            break
    else:
        print(f"KMeans reached max_iter={max_iter}; last changes={changes}; max shift={center_shift.max():.4f}")
    label_map = labels.reshape((h, w))
    means = np.clip(np.round(centers), 0, 255).astype(np.uint8)
    recon = means[labels].reshape((h, w, 3)).astype(np.uint8)
    return label_map, means, recon

def compute_gradient_gray(image_bgr):
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad = np.sqrt(gx * gx + gy * gy)
    return grad

def slic_superpixels(image_bgr, num_superpixels=200, compactness=10.0, max_iter=10):
    h, w, _ = image_bgr.shape
    N = h * w
    S = int(math.sqrt(N / float(num_superpixels)) + 0.5)
    if S < 1:
        S = 1
    img_lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    centers = []  
    for y in range(S // 2, h, S):
        for x in range(S // 2, w, S):
            centers.append([img_lab[y, x, 0], img_lab[y, x, 1], img_lab[y, x, 2], float(y), float(x)])
    centers = np.array(centers, dtype=np.float32)
    K = centers.shape[0]
    grad = compute_gradient_gray(image_bgr)
    for i in range(K):
        y = int(round(centers[i, 3]))
        x = int(round(centers[i, 4]))
        min_g = grad[y, x]
        best = (y, x)
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                yy = min(max(y + dy, 0), h - 1)
                xx = min(max(x + dx, 0), w - 1)
                if grad[yy, xx] < min_g:
                    min_g = grad[yy, xx]
                    best = (yy, xx)
        by, bx = best
        centers[i, :3] = img_lab[by, bx]
        centers[i, 3] = float(by)
        centers[i, 4] = float(bx)
    labels = -1 * np.ones((h, w), dtype=np.int32)
    distances = 1e12 * np.ones((h, w), dtype=np.float32)
    for it in range(max_iter):
        for ci in range(K):
            Lc, Ac, Bc, yc, xc = centers[ci]
            y0 = int(max(yc - S, 0))
            y1 = int(min(yc + S, h - 1))
            x0 = int(max(xc - S, 0))
            x1 = int(min(xc + S, w - 1))
            window_lab = img_lab[y0:y1+1, x0:x1+1]
            yy, xx = np.mgrid[y0:y1+1, x0:x1+1]
            color_dist2 = np.sum((window_lab - centers[ci, :3]) ** 2, axis=2)
            spatial_dist2 = (yy - yc) ** 2 + (xx - xc) ** 2
            D = np.sqrt(color_dist2) + (compactness / float(S)) * np.sqrt(spatial_dist2)
            local_distances = distances[y0:y1+1, x0:x1+1]
            local_labels = labels[y0:y1+1, x0:x1+1]
            mask = D < local_distances
            local_distances[mask] = D[mask]
            local_labels[mask] = ci
            distances[y0:y1+1, x0:x1+1] = local_distances
            labels[y0:y1+1, x0:x1+1] = local_labels
        new_centers = np.zeros_like(centers)
        counts = np.zeros((K,), dtype=np.int32)
        for ci in range(K):
            mask = (labels == ci)
            cnt = mask.sum()
            if cnt == 0:
                new_centers[ci] = centers[ci]
                continue
            counts[ci] = cnt
            new_centers[ci, 0:3] = img_lab[mask].mean(axis=0)
            ys, xs = np.nonzero(mask)
            new_centers[ci, 3] = ys.mean()
            new_centers[ci, 4] = xs.mean()
        shift = np.sqrt(np.sum((new_centers - centers) ** 2, axis=1))
        centers = new_centers
        distances.fill(1e12)
        if shift.max() < 0.5:
            print(f"SLIC converged at iteration {it}; max center shift={shift.max():.4f}")
            break
    label_img = labels.copy()
    new_label = 0
    final_labels = -1 * np.ones_like(label_img)
    visited = np.zeros_like(label_img, dtype=bool)
    min_size = int(N / (4 * K))  
    for y in range(h):
        for x in range(w):
            if visited[y, x]:
                continue
            cur_label = label_img[y, x]
            stack = [(y, x)]
            comp_coords = []
            while stack:
                yy, xx = stack.pop()
                if yy < 0 or yy >= h or xx < 0 or xx >= w:
                    continue
                if visited[yy, xx]:
                    continue
                if label_img[yy, xx] != cur_label:
                    continue
                visited[yy, xx] = True
                comp_coords.append((yy, xx))
                stack.append((yy+1, xx))
                stack.append((yy-1, xx))
                stack.append((yy, xx+1))
                stack.append((yy, xx-1))
            if len(comp_coords) <= min_size:
                for (yy, xx) in comp_coords:
                    found = False
                    for dy, dx in ((1,0),(-1,0),(0,1),(0,-1)):
                        ny, nx = yy+dy, xx+dx
                        if 0 <= ny < h and 0 <= nx < w and label_img[ny, nx] != cur_label:
                            final_labels[yy, xx] = label_img[ny, nx]
                            found = True
                            break
                    if not found:
                        final_labels[yy, xx] = cur_label
            else:
                for (yy, xx) in comp_coords:
                    final_labels[yy, xx] = new_label
                new_label += 1
    if final_labels.min() < 0:
        mask = final_labels < 0
        final_labels[mask] = label_img[mask]
    unique = np.unique(final_labels)
    mapping = {v:i for i,v in enumerate(unique)}
    compact = np.vectorize(lambda v: mapping[int(v)])(final_labels).astype(np.int32)
    M = compact.max() + 1
    out_img = np.zeros_like(image_bgr, dtype=np.uint8)
    for si in range(M):
        mask = (compact == si)
        if mask.sum() == 0:
            continue
        mean_col = image_bgr[mask].mean(axis=0)
        out_img[mask] = np.clip(np.round(mean_col), 0, 255).astype(np.uint8)
    return compact, out_img

def rms_gradient_rgb(image_bgr):
    grads = []
    for c in range(3):
        channel = image_bgr[:, :, c].astype(np.float32)
        gx = cv2.Sobel(channel, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(channel, cv2.CV_32F, 0, 1, ksize=3)
        mag = np.sqrt(gx * gx + gy * gy)
        grads.append(mag)
    grads = np.stack(grads, axis=2)
    rms = np.sqrt(np.mean(grads ** 2, axis=2))
    return rms

def watershed_segmentation(image_bgr, smooth_sigma=1.0):
    h, w, _ = image_bgr.shape
    ksize = int(6 * smooth_sigma + 1)
    if ksize % 2 == 0:
        ksize += 1
    smooth = cv2.GaussianBlur(image_bgr, (ksize, ksize), smooth_sigma)
    grad = rms_gradient_rgb(smooth)
    gray = cv2.cvtColor(smooth, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)
    sure_bg = cv2.dilate(opening, kernel, iterations=3)
    dist = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    dt_max = dist.max() if dist.max() > 0 else 1.0
    _, sure_fg = cv2.threshold(dist, 0.4 * dt_max, 255, 0)
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)
    num_markers, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0
    img_for_ws = smooth.copy()
    cv2.watershed(img_for_ws, markers)
    labels = markers.copy()
    unique = np.unique(labels)
    rng = np.random.RandomState(12345)
    color_map = {}
    for lab in unique:
        if lab == -1 or lab == 1:
            color_map[int(lab)] = (0, 0, 0)
        else:
            color_map[int(lab)] = tuple(int(x) for x in rng.randint(0, 256, 3))
    out = np.zeros((h, w, 3), dtype=np.uint8)
    for y in range(h):
        for x in range(w):
            lab = int(labels[y, x])
            out[y, x] = color_map.get(lab, (0, 0, 0))
    boundaries = (labels == -1)
    overlay = image_bgr.copy()
    overlay[boundaries] = np.array((0, 0, 255), dtype=np.uint8)  
    return labels, out, overlay

def part_a_main(img_path, out_dir, k=5, auto_seed = True):
    print("======Part-A processing...======")
    img = cv2.imread(img_path)
    seeds = None
    if not(auto_seed):
        seeds = get_manual_seeds(img, k)
    if seeds is not None:
        init_points = seeds
    else:
        init_points = None
    labels, means, recon = kmeans_rgb_segmentation(img, k=k, max_iter=200, tol=0.5, init_points=init_points)
    bmask = highlight_boundaries(labels)
    recon_bgr = recon.copy()
    recon_bgr[bmask] = np.array([0, 0, 0], dtype=np.uint8)  
    ensure_dir(out_dir)
    save_img(os.path.join(out_dir, 'PartA', 'kmeans_reconstruction.png'), recon_bgr)
    h, w = labels.shape
    label_vis = np.zeros((h, w, 3), dtype=np.uint8)
    rng = np.random.RandomState(0)
    unique = np.unique(labels)
    color_map = {lab: tuple(int(x) for x in rng.randint(0,256,3)) for lab in unique}
    for lab in unique:
        label_vis[labels == lab] = color_map[lab]
    save_img(os.path.join(out_dir, 'PartA', 'kmeans_labels.png'), label_vis)
    print(f"Part A outputs written to {os.path.join(out_dir, 'PartA')}")

def part_b_main(img_path, out_dir, num_superpixels=200, compactness=15.0):
    print("======Part-B processing...======")
    img = cv2.imread(img_path)
    labels, out_img = slic_superpixels(img, num_superpixels=num_superpixels, compactness=compactness, max_iter=10)
    bmask = highlight_boundaries(labels)
    out_overlay = out_img.copy()
    out_overlay[bmask] = np.array([0,0,0], dtype=np.uint8)
    ensure_dir(os.path.join(out_dir, 'PartB'))
    save_img(os.path.join(out_dir, 'PartB', 'slic_superpixels.png'), out_img)
    save_img(os.path.join(out_dir, 'PartB', 'slic_superpixels_boundaries.png'), out_overlay)
    print(f"Part B outputs written to {os.path.join(out_dir, 'PartB')}")

def part_c_main(img_path, out_dir, smooth_sigma=1.0):
    print("======Part-C processing...======")
    img = cv2.imread(img_path)
    labels, colored_basins, overlay = watershed_segmentation(img, smooth_sigma=smooth_sigma)
    ensure_dir(os.path.join(out_dir, 'PartC'))
    save_img(os.path.join(out_dir, 'PartC', 'watershed_basins.png'), colored_basins)
    save_img(os.path.join(out_dir, 'PartC', 'watershed_overlay_boundaries.png'), overlay)
    print(f"Part C outputs written to {os.path.join(out_dir, 'PartC')}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='../Testcases/q5_3.jpg')
    parser.add_argument('--out', type=str, default='./Q5_Output')
    parser.add_argument('--part', type=str, choices=['A','B','C','all'], default='all')
    parser.add_argument('--k', type=int, default=8)
    parser.add_argument('--auto_seed', type=bool, default=True)
    parser.add_argument('--num_superpixels', type=int, default=200)
    parser.add_argument('--compactness', type=float, default=15.0)
    parser.add_argument('--smooth_sigma', type=float, default=1.0)
    args = parser.parse_args()
    ensure_dir(args.out)
    ensure_dir(os.path.join(args.out, 'PartA'))
    ensure_dir(os.path.join(args.out, 'PartB'))
    ensure_dir(os.path.join(args.out, 'PartC'))
    if args.part in ('A', 'all'):
        part_a_main(args.input, args.out, k=args.k, auto_seed=args.auto_seed)
    if args.part in ('B', 'all'):
        part_b_main(args.input, args.out, num_superpixels=args.num_superpixels, compactness=args.compactness)
    if args.part in ('C', 'all'):
        part_c_main(args.input, args.out, smooth_sigma=args.smooth_sigma)

if __name__ == '__main__':
    main()

