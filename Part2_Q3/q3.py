import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
INPUT_IMAGE = '../Testcases/q3_input.tif'
OUTPUT_DIR = './Q3_Output'
PARTA_DIR = os.path.join(OUTPUT_DIR, 'PartA')
PARTB_DIR = os.path.join(OUTPUT_DIR, 'PartB')
PARTC_DIR = os.path.join(OUTPUT_DIR, 'PartC')
def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
def save_image(img, path, cmap='gray'):
    if len(img.shape) == 2:
        plt.imsave(path, img, cmap=cmap)
    else:
        plt.imsave(path, cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
def part_a_canny_edge_detection(image_path, output_dir, gaussian_ksize=5, sigma=1.4, high_thresh=60, low_thresh=30):
    print("======Part-A processing...======")
    ensure_dir(output_dir)
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img_blur = cv2.GaussianBlur(img, (gaussian_ksize, gaussian_ksize), sigma)
    grad_x = cv2.Sobel(img_blur, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(img_blur, cv2.CV_64F, 0, 1, ksize=3)
    mag = np.sqrt(grad_x ** 2 + grad_y ** 2)
    mag = np.uint8(255 * mag / mag.max())
    save_image(mag, os.path.join(output_dir, 'gradient_magnitude.png'))
    def non_max_suppression(mag, grad_x, grad_y):
        angle = np.arctan2(grad_y, grad_x) * 180 / np.pi
        angle[angle < 0] += 180
        nms = np.zeros_like(mag)
        rows, cols = mag.shape
        for i in range(1, rows - 1):
            for j in range(1, cols - 1):
                q, r = 255, 255
                if (0 <= angle[i,j] < 22.5) or (157.5 <= angle[i,j] < 180):
                    q = mag[i, j+1]
                    r = mag[i, j-1]
                elif (22.5 <= angle[i,j] < 67.5):
                    q = mag[i+1, j-1]
                    r = mag[i-1, j+1]
                elif (67.5 <= angle[i,j] < 112.5):
                    q = mag[i+1, j]
                    r = mag[i-1, j]
                elif (112.5 <= angle[i,j] < 157.5):
                    q = mag[i-1, j-1]
                    r = mag[i+1, j+1]
                if mag[i,j] >= q and mag[i,j] >= r:
                    nms[i,j] = mag[i,j]
        return nms
    nms_img = non_max_suppression(mag, grad_x, grad_y)
    save_image(nms_img, os.path.join(output_dir, 'non_max_suppression.png'))
    high_thresh_img = np.where(nms_img >= high_thresh, nms_img, 0).astype(np.uint8)
    save_image(high_thresh_img, os.path.join(output_dir, 'high_threshold.png'))
    low_thresh_img = np.where(nms_img >= low_thresh, nms_img, 0).astype(np.uint8)
    save_image(low_thresh_img, os.path.join(output_dir, 'low_threshold.png'))
    canny_result = cv2.Canny(img_blur, low_thresh, high_thresh)
    save_image(canny_result, os.path.join(output_dir, 'canny_edges.png'))
    return canny_result
def part_b_hough_transform(edge_img, output_dir, rho_bin=1, theta_bin=np.pi/180):
    print("======Part-B processing...======")
    ensure_dir(output_dir)
    h, w = edge_img.shape
    diag_len = int(np.ceil(np.sqrt(h**2 + w**2)))
    rhos = np.arange(-diag_len, diag_len + 1, rho_bin)
    thetas = np.arange(0, np.pi, theta_bin)
    accumulator = np.zeros((len(rhos), len(thetas)), dtype=np.uint64)
    y_idxs, x_idxs = np.nonzero(edge_img)
    for i in range(len(x_idxs)):
        x = x_idxs[i]
        y = y_idxs[i]
        for t_idx in range(len(thetas)):
            rho = int(round(x * np.cos(thetas[t_idx]) + y * np.sin(thetas[t_idx]))) + diag_len
            accumulator[rho, t_idx] += 1
    hough_img = np.uint8(255 * accumulator / accumulator.max())
    save_image(hough_img, os.path.join(output_dir, 'hough_transform.png'))
    print(f"Hough bin widths used: rho = {rho_bin}, theta = {theta_bin}")
    print(f"Accumulator array size: {accumulator.shape}")
    return accumulator, rhos, thetas
def part_c_detect_lines_and_intersections(accumulator, rhos, thetas, orig_img_path, output_dir, k=10, nms_dist=15, nms_angle=15):
    print("======Part-C processing...======")
    ensure_dir(output_dir)
    img = cv2.imread(orig_img_path)
    h, w = img.shape[:2]
    acc = accumulator.copy()
    lines = []
    for _ in range(k):
        idx = np.unravel_index(np.argmax(acc), acc.shape)
        rho_idx, theta_idx = idx
        if acc[rho_idx, theta_idx] == 0:
            break
        lines.append((rhos[rho_idx], thetas[theta_idx]))
        min_r = max(0, rho_idx - nms_dist)
        max_r = min(acc.shape[0], rho_idx + nms_dist)
        min_t = max(0, theta_idx - nms_angle)
        max_t = min(acc.shape[1], theta_idx + nms_angle)
        acc[min_r:max_r, min_t:max_t] = 0
    img_lines = img.copy()
    for rho, theta in lines:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        pt1 = (int(x0 + 1000 * (-b)), int(y0 + 1000 * a))
        pt2 = (int(x0 - 1000 * (-b)), int(y0 - 1000 * a))
        cv2.line(img_lines, pt1, pt2, (0, 0, 255), 2)
    save_image(img_lines, os.path.join(output_dir, 'top_lines.png'), cmap=None)
    intersections = []
    eps = 1e-9
    for i in range(len(lines)):
        for j in range(i+1, len(lines)):
            rho1, theta1 = lines[i]
            rho2, theta2 = lines[j]
            denom = np.cos(theta1) * np.sin(theta2) - np.sin(theta1) * np.cos(theta2)
            if np.abs(denom) < eps:
                continue
            x = (rho1 * np.sin(theta2) - rho2 * np.sin(theta1)) / denom
            y = (np.cos(theta1) * rho2 - np.cos(theta2) * rho1) / denom
            x_i = int(round(x))
            y_i = int(round(y))
            if 0 <= x_i < w and 0 <= y_i < h:
                intersections.append((x_i, y_i))
    img_intersections = img_lines.copy()
    for (x, y) in intersections:
        cv2.circle(img_intersections, (x, y), 6, (0, 255, 0), -1)
    save_image(img_intersections, os.path.join(output_dir, 'lines_and_intersections.png'), cmap=None)
def main():
    edge_img = part_a_canny_edge_detection(INPUT_IMAGE, PARTA_DIR)
    accumulator, rhos, thetas = part_b_hough_transform(edge_img, PARTB_DIR)
    part_c_detect_lines_and_intersections(accumulator, rhos, thetas, INPUT_IMAGE, PARTC_DIR, k=10)
if __name__ == "__main__":
    main()

