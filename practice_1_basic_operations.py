import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import rgb_to_hsv, hsv_to_rgb
import os

def practice_1():
    print("=== Practice 1: Basic Image Operations ===\n")
    
    print("1. Reading, Displaying and Saving Image Files")
    
    img = cv2.imread('sample_img.png')
    if img is None:
        print("Error: Could not load sample_img.png. Please make sure the file exists.")
        return
    
    print(f"✓ Image loaded successfully. Shape: {img.shape}")
    
    plt.figure(figsize=(15, 10))
    plt.subplot(2, 4, 1)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title('Original Image')
    plt.axis('off')
    
    print("\n2. Resizing Image")
    
    height, width = img.shape[:2]
    img_larger = cv2.resize(img, (width * 2, height * 2), interpolation=cv2.INTER_CUBIC)
    print(f"✓ Larger image created. Original: {img.shape[:2]}, New: {img_larger.shape[:2]}")
    
    img_smaller = cv2.resize(img, (width // 2, height // 2), interpolation=cv2.INTER_AREA)
    print(f"✓ Smaller image created. Original: {img.shape[:2]}, New: {img_smaller.shape[:2]}")
    
    plt.subplot(2, 4, 2)
    plt.imshow(cv2.cvtColor(img_smaller, cv2.COLOR_BGR2RGB))
    plt.title('Resized (Smaller)')
    plt.axis('off')
    
    print("\n3. Flipping Image")
    
    img_horizontal = cv2.flip(img, 1)
    print("✓ Horizontal flip completed")
    
    img_vertical = cv2.flip(img, 0)
    print("✓ Vertical flip completed")
    
    img_both = cv2.flip(img, -1)
    print("✓ Both horizontal and vertical flip completed")
    
    plt.subplot(2, 4, 3)
    plt.imshow(cv2.cvtColor(img_horizontal, cv2.COLOR_BGR2RGB))
    plt.title('Horizontal Flip')
    plt.axis('off')
    
    print("\n4. Cropping Image by 33%")
    
    crop_size = int(min(height, width) * 0.33)
    start_row = (height - crop_size) // 2
    start_col = (width - crop_size) // 2
    img_cropped = img[start_row:start_row + crop_size, start_col:start_col + crop_size]
    print(f"✓ Image cropped. Original: {img.shape[:2]}, Cropped: {img_cropped.shape[:2]}")
    
    plt.subplot(2, 4, 4)
    plt.imshow(cv2.cvtColor(img_cropped, cv2.COLOR_BGR2RGB))
    plt.title('Cropped (33%)')
    plt.axis('off')
    
    print("\n5. Switching between color spaces")
    
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    print("✓ Converted to Grayscale")
    
    print("✓ BGR format (original)")
    
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    print("✓ Converted to HSV")
    
    img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    print("✓ Converted to LAB")
    
    img_xyz = cv2.cvtColor(img, cv2.COLOR_BGR2XYZ)
    print("✓ Converted to XYZ")
    
    plt.subplot(2, 4, 5)
    plt.imshow(img_gray, cmap='gray')
    plt.title('Grayscale')
    plt.axis('off')
    
    plt.subplot(2, 4, 6)
    plt.imshow(cv2.cvtColor(img_hsv, cv2.COLOR_HSV2RGB))
    plt.title('HSV')
    plt.axis('off')

    # Add LAB and XYZ visualization
    plt.subplot(2, 4, 7)
    plt.imshow(cv2.cvtColor(img_lab, cv2.COLOR_LAB2RGB))
    plt.title('LAB')
    plt.axis('off')

    plt.subplot(2, 4, 8)
    plt.imshow(cv2.cvtColor(img_xyz, cv2.COLOR_XYZ2RGB))
    plt.title('XYZ')
    plt.axis('off')

    plt.tight_layout()
    plt.show()
    
    print("\n6. Splitting and Merging Color Channels")
    
    b_channel, g_channel, r_channel = cv2.split(img)
    print("✓ Color channels split successfully")
    
    zeros = np.zeros_like(b_channel)
    b_colored = cv2.merge([b_channel, zeros, zeros])
    g_colored = cv2.merge([zeros, g_channel, zeros])
    r_colored = cv2.merge([zeros, zeros, r_channel])
    
    img_merged = cv2.merge([b_channel, g_channel, r_channel])
    print("✓ Color channels merged successfully")
    
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 4, 1)
    plt.imshow(cv2.cvtColor(b_colored, cv2.COLOR_BGR2RGB))
    plt.title('Blue Channel')
    plt.axis('off')
    
    plt.subplot(1, 4, 2)
    plt.imshow(cv2.cvtColor(g_colored, cv2.COLOR_BGR2RGB))
    plt.title('Green Channel')
    plt.axis('off')
    
    plt.subplot(1, 4, 3)
    plt.imshow(cv2.cvtColor(r_colored, cv2.COLOR_BGR2RGB))
    plt.title('Red Channel')
    plt.axis('off')
    
    plt.subplot(1, 4, 4)
    plt.imshow(cv2.cvtColor(img_merged, cv2.COLOR_BGR2RGB))
    plt.title('Merged Image')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    print("\n7. Grayscale Histogram and Equalization")
    
    hist_gray = cv2.calcHist([img_gray], [0], None, [256], [0, 256])
    
    img_eq_gray = cv2.equalizeHist(img_gray)
    hist_eq_gray = cv2.calcHist([img_eq_gray], [0], None, [256], [0, 256])
    
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 3, 1)
    plt.imshow(img_gray, cmap='gray')
    plt.title('Original Grayscale')
    plt.axis('off')
    
    plt.subplot(2, 3, 2)
    plt.plot(hist_gray)
    plt.title('Original Histogram')
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Frequency')
    
    plt.subplot(2, 3, 3)
    plt.imshow(img_eq_gray, cmap='gray')
    plt.title('Histogram Equalized')
    plt.axis('off')
    
    plt.subplot(2, 3, 4)
    plt.plot(hist_eq_gray)
    plt.title('Equalized Histogram')
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Frequency')
    
    print("✓ Grayscale histogram analysis completed")
    
    print("\n8. Color Histogram and Equalization")
    
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    colors = ['red', 'green', 'blue']
    plt.subplot(2, 3, 5)
    for i, color in enumerate(colors):
        hist = cv2.calcHist([img], [i], None, [256], [0, 256])
        plt.plot(hist, color=color, alpha=0.7)
    plt.title('Color Histogram')
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Frequency')
    plt.legend(colors)
    
    img_lab_eq = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    img_lab_eq[:,:,0] = cv2.equalizeHist(img_lab_eq[:,:,0])
    img_eq_color = cv2.cvtColor(img_lab_eq, cv2.COLOR_LAB2BGR)
    img_eq_color_rgb = cv2.cvtColor(img_eq_color, cv2.COLOR_BGR2RGB)
    
    plt.subplot(2, 3, 6)
    plt.imshow(img_eq_color_rgb)
    plt.title('Color Histogram Equalized')
    plt.axis('off')
    
    print("✓ Color histogram analysis completed")
    
    plt.tight_layout()
    plt.show()
    
    print("\n✓ All image processing operations completed!")
    print("\n=== Practice 1 Completed ===")

if __name__ == "__main__":
    practice_1() 