import numpy as np
from PIL import Image
import os
def custom_erosion(image, se):
    image = image.astype(np.uint8)
    se = se.astype(np.uint8)
    output = np.zeros_like(image)
    se_h, se_w = se.shape
    cen_y = se_h // 2
    cen_x = se_w // 2
    padded = np.pad(image, ((cen_y, cen_y), (cen_x, cen_x)), mode='constant', constant_values=0)
    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            neigh = padded[y:y+se_h, x:x+se_w]
            masked = neigh * se
            if np.all(masked == se):
                output[y, x] = 1
    return output
def custom_dilation(image, se):
    image = image.astype(np.uint8)
    se = se.astype(np.uint8)
    se = np.flip(se)
    output = np.zeros_like(image)
    se_h, se_w = se.shape
    cen_y = se_h // 2
    cen_x = se_w // 2
    padded = np.pad(image, ((cen_y, cen_y), (cen_x, cen_x)), mode='constant', constant_values=0)
    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            neigh = padded[y:y+se_h, x:x+se_w]
            masked = neigh * se
            if np.any(masked > 0):
                output[y, x] = 1
    return output
def custom_opening(image, se):
    eroded = custom_erosion(image, se)
    opened = custom_dilation(eroded, se)
    return opened
def custom_closing(image, se):
    dilated = custom_dilation(image, se)
    closed = custom_erosion(dilated, se)
    return closed
def create_disk_se(radius):
    y, x = np.ogrid[-radius:radius+1, -radius:radius+1]
    disk = x**2 + y**2 <= radius**2
    return disk.astype(np.uint8)
def hit_or_miss(image, se1, se2):
    eroded_fg = custom_erosion(image, se1)
    eroded_bg = custom_erosion(1 - image, se2)
    result = eroded_fg * eroded_bg
    return result
def create_rgb_overlay(original, detected):
    rgb = np.zeros((*original.shape, 3), dtype=np.uint8)
    rgb[:, :, 0] = original * 255  
    rgb[:, :, 1] = detected * 255  
    rgb[:, :, 2] = detected * 255  
    return rgb
def save_image(image, path):
    if len(image.shape) == 2:
        img = Image.fromarray((image * 255).astype(np.uint8))
    else:
        img = Image.fromarray(image.astype(np.uint8))
    img.save(path)
    print(f"Saved: {path}")
def main():
    """Main function to execute all parts of the assignment."""
    os.makedirs('./Q1_Output', exist_ok=True)
    os.makedirs('./Q1_Output/PartA', exist_ok=True)
    os.makedirs('./Q1_Output/PartB', exist_ok=True)
    os.makedirs('./Q1_Output/PartC', exist_ok=True)
    os.makedirs('./Q1_Output/PartD', exist_ok=True)
    print("=" * 50)
    print("======Part-A processing...======")
    print("=" * 50)
    text_img = Image.open('../Testcases/q1_text.jpg').convert('L')
    text_array = np.array(text_img)
    print(f"Loaded text image: shape {text_array.shape}")
    threshold = 128
    A = (text_array < threshold).astype(np.uint8)
    print(f"Binary image A created: {A.shape}, white pixels: {np.sum(A)}")
    save_image(A, './Q1_Output/PartA/binary_image_A.png')
    disk_radius = 3
    disk_se = create_disk_se(disk_radius)
    print(f"Created disk SE with radius {disk_radius}: shape {disk_se.shape}")
    save_image(disk_se, './Q1_Output/PartA/disk_se.png')
    print("Performing dilation...")
    A_dilated = custom_dilation(A, disk_se)
    save_image(A_dilated, './Q1_Output/PartA/A_dilated.png')
    print(f"Dilation complete: white pixels = {np.sum(A_dilated)}")
    print("Performing closing...")
    A_closed = custom_closing(A, disk_se)
    save_image(A_closed, './Q1_Output/PartA/A_closed.png')
    print(f"Closing complete: white pixels = {np.sum(A_closed)}")
    print("\nPart A completed successfully!")
    print("\n" + "=" * 50)
    print("======Part-B processing...======")
    print("=" * 50)
    letter_c_img = Image.open('../Testcases/q1_c_letter.jpg').convert('L')
    letter_c_array = np.array(letter_c_img)
    print(f"Loaded letter 'c' image: shape {letter_c_array.shape}")
    B = (letter_c_array < 128).astype(np.uint8)
    print(f"Binary SE B created: {B.shape}, foreground pixels: {np.sum(B)}")
    save_image(B, './Q1_Output/PartB/SE_B_original.png')
    print("Performing opening operation with B on A...")
    A_opened_B = custom_opening(A, B)
    save_image(A_opened_B, './Q1_Output/PartB/A_opened_with_B.png')
    print(f"Opening complete: detected pixels = {np.sum(A_opened_B)}")
    rgb_result_B = create_rgb_overlay(A, A_opened_B)
    save_image(rgb_result_B, './Q1_Output/PartB/result_overlay_B.png')
    print("RGB overlay created")
    print("\nPart B completed successfully!")
    print("Note: Not all c's are found because the SE must match exactly")
    print("\n" + "=" * 50)
    print("======Part-C processing...======")
    print("=" * 50)
    print("Creating modified SE B1 by eroding B...")
    small_se = np.ones((2, 2), dtype=np.uint8)
    B1 = custom_erosion(B, small_se)
    if np.sum(B1) < 3:
        print("Using alternative erosion approach...")
        cross_se = np.array([[0, 1, 0],
                            [1, 1, 1],
                            [0, 1, 0]], dtype=np.uint8)
        B1 = custom_erosion(B, cross_se)
    if np.sum(B1) < 3:
        print("Using minimal erosion approach...")
        B1 = B.copy()
        B1[0, :] = 0
        B1[-1, :] = 0
        B1[:, 0] = 0
        B1[:, -1] = 0
    print(f"Modified SE B1 created: {B1.shape}, foreground pixels: {np.sum(B1)}")
    save_image(B1, './Q1_Output/PartC/SE_B1_modified.png')
    print("Performing opening operation with B1 on A...")
    A_opened_B1 = custom_opening(A, B1)
    save_image(A_opened_B1, './Q1_Output/PartC/A_opened_with_B1.png')
    print(f"Opening complete: detected pixels = {np.sum(A_opened_B1)}")
    rgb_result_C = create_rgb_overlay(A, A_opened_B1)
    save_image(rgb_result_C, './Q1_Output/PartC/result_overlay_C.png')
    print("RGB overlay created")
    print("\nPart C completed successfully!")
    print("B1 was obtained by eroding B to make it more flexible")
    print("\n" + "=" * 50)
    print("======Part-D processing...======")
    print("=" * 50)
    print("Creating SE B2 for background matching...")
    dilate_se = np.ones((3, 3), dtype=np.uint8)
    B_dilated = custom_dilation(B, dilate_se)
    B2 = B_dilated.copy()
    B2[B > 0] = 0
    width_B = B2.shape[1]
    height_B = B2.shape[0]
    B2[:, :width_B//2] = 0
    B2[:height_B//3, :] = 0
    B2[2*height_B//3:, :] = 0
    print(f"SE B2 created: {B2.shape}, background pixels: {np.sum(B2)}")
    save_image(B2, './Q1_Output/PartD/SE_B2_background.png')
    print("Performing hit-or-miss transform with B1 and B2...")
    A_hit_miss = hit_or_miss(A, B1, B2)
    save_image(A_hit_miss, './Q1_Output/PartD/A_hit_miss_result.png')
    print(f"Hit-or-miss complete: detected points = {np.sum(A_hit_miss)}")
    A_detected_letters = custom_dilation(A_hit_miss, B)
    rgb_result_D = create_rgb_overlay(A, A_detected_letters)
    save_image(rgb_result_D, './Q1_Output/PartD/result_overlay_D.png')
    rgb_points_D = create_rgb_overlay(A, A_hit_miss)
    save_image(rgb_points_D, './Q1_Output/PartD/result_points_overlay_D.png')
    print("RGB overlays created")
    print("\nPart D completed successfully!")
    print("B2 matches the background in the opening region of 'c'")
    print("\n" + "=" * 60)
    print("ALL PARTS COMPLETED SUCCESSFULLY")
    print("=" * 60)
    print("\nAll results saved in ./Q1_Output/ subdirectories")
    print("Check PartA, PartB, PartC, and PartD folders for outputs")
    print("=" * 60)
if __name__ == "__main__":
    main()

