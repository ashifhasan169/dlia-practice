import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def practice_2():
    print("=== Practice 2: Advanced Image Operations ===\n")
    
    img = cv2.imread('sample_img.png')
    if img is None:
        print("Error: Could not load sample_img.png. Please make sure the file exists.")
        return
    
    print("✓ Sample image loaded")
    
    img1 = cv2.resize(img, (200, 200))
    img2 = cv2.resize(img, (200, 200))
    
    binary1 = np.zeros((150, 150), dtype=np.uint8)
    cv2.rectangle(binary1, (30, 30), (120, 120), 255, -1)
    
    binary2 = np.zeros((150, 150), dtype=np.uint8)
    cv2.circle(binary2, (75, 75), 50, 255, -1)
    
    complex_img = cv2.resize(img, (300, 300))
    
    plt.figure(figsize=(20, 15))
    
    print("1. Arithmetic Operations on Images")
    
    img_addition = cv2.add(img1, img2)
    print("✓ Image addition completed")
    
    img_subtraction = cv2.subtract(img1, img2)
    print("✓ Image subtraction completed")
    
    img_blended = cv2.addWeighted(img1, 0.7, img2, 0.3, 0)
    print("✓ Image blending completed")
    
    plt.subplot(4, 5, 1)
    plt.imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
    plt.title('Image 1')
    plt.axis('off')
    
    plt.subplot(4, 5, 2)
    plt.imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
    plt.title('Image 2')
    plt.axis('off')
    
    plt.subplot(4, 5, 3)
    plt.imshow(cv2.cvtColor(img_addition, cv2.COLOR_BGR2RGB))
    plt.title('Addition')
    plt.axis('off')
    
    plt.subplot(4, 5, 4)
    plt.imshow(cv2.cvtColor(img_subtraction, cv2.COLOR_BGR2RGB))
    plt.title('Subtraction')
    plt.axis('off')
    
    plt.subplot(4, 5, 5)
    plt.imshow(cv2.cvtColor(img_blended, cv2.COLOR_BGR2RGB))
    plt.title('Blended (0.7 + 0.3)')
    plt.axis('off')
    
    print("\n2. Bitwise Operations on Binary Images")
    
    img_and = cv2.bitwise_and(binary1, binary2)
    print("✓ Bitwise AND completed")
    
    img_or = cv2.bitwise_or(binary1, binary2)
    print("✓ Bitwise OR completed")
    
    img_xor = cv2.bitwise_xor(binary1, binary2)
    print("✓ Bitwise XOR completed")
    
    img_not = cv2.bitwise_not(binary1)
    print("✓ Bitwise NOT completed")
    
    plt.subplot(4, 5, 6)
    plt.imshow(binary1, cmap='gray')
    plt.title('Binary Image 1')
    plt.axis('off')
    
    plt.subplot(4, 5, 7)
    plt.imshow(binary2, cmap='gray')
    plt.title('Binary Image 2')
    plt.axis('off')
    
    plt.subplot(4, 5, 8)
    plt.imshow(img_and, cmap='gray')
    plt.title('AND Operation')
    plt.axis('off')
    
    plt.subplot(4, 5, 9)
    plt.imshow(img_or, cmap='gray')
    plt.title('OR Operation')
    plt.axis('off')
    
    plt.subplot(4, 5, 10)
    plt.imshow(img_xor, cmap='gray')
    plt.title('XOR Operation')
    plt.axis('off')
    
    complex_gray = cv2.cvtColor(complex_img, cv2.COLOR_BGR2GRAY)
    
    print("\n3. Simple Thresholding")
    
    ret, thresh_binary = cv2.threshold(complex_gray, 127, 255, cv2.THRESH_BINARY)
    ret, thresh_binary_inv = cv2.threshold(complex_gray, 127, 255, cv2.THRESH_BINARY_INV)
    ret, thresh_trunc = cv2.threshold(complex_gray, 127, 255, cv2.THRESH_TRUNC)
    ret, thresh_tozero = cv2.threshold(complex_gray, 127, 255, cv2.THRESH_TOZERO)
    
    print(f"✓ Simple thresholding completed with threshold value: {ret}")
    
    plt.subplot(4, 5, 11)
    plt.imshow(complex_gray, cmap='gray')
    plt.title('Original Grayscale')
    plt.axis('off')
    
    plt.subplot(4, 5, 12)
    plt.imshow(thresh_binary, cmap='gray')
    plt.title('Binary Threshold')
    plt.axis('off')
    
    plt.subplot(4, 5, 13)
    plt.imshow(thresh_binary_inv, cmap='gray')
    plt.title('Binary Inv Threshold')
    plt.axis('off')
    
    print("\n4. Adaptive Thresholding")
    
    adaptive_mean = cv2.adaptiveThreshold(complex_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
                                         cv2.THRESH_BINARY, 11, 2)
    adaptive_gaussian = cv2.adaptiveThreshold(complex_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                             cv2.THRESH_BINARY, 11, 2)
    
    print("✓ Adaptive thresholding completed")
    
    plt.subplot(4, 5, 14)
    plt.imshow(adaptive_mean, cmap='gray')
    plt.title('Adaptive Mean')
    plt.axis('off')
    
    plt.subplot(4, 5, 15)
    plt.imshow(adaptive_gaussian, cmap='gray')
    plt.title('Adaptive Gaussian')
    plt.axis('off')
    
    print("\n5. Otsu Thresholding")
    
    ret_otsu, thresh_otsu = cv2.threshold(complex_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    print(f"✓ Otsu thresholding completed with optimal threshold: {ret_otsu:.2f}")
    
    plt.subplot(4, 5, 16)
    plt.imshow(thresh_otsu, cmap='gray')
    plt.title(f'Otsu Threshold ({ret_otsu:.0f})')
    plt.axis('off')
    
    print("\n6. Segmentation using Thresholding")
    
    segmentation_img = complex_gray
    
    _, seg1 = cv2.threshold(segmentation_img, 85, 255, cv2.THRESH_BINARY)
    _, seg2 = cv2.threshold(segmentation_img, 170, 255, cv2.THRESH_BINARY)
    
    segmented = np.zeros_like(segmentation_img)
    segmented[segmentation_img < 85] = 50
    segmented[(segmentation_img >= 85) & (segmentation_img < 170)] = 150
    segmented[segmentation_img >= 170] = 255
    
    print("✓ Image segmentation completed")
    
    plt.subplot(4, 5, 17)
    plt.imshow(segmented, cmap='gray')
    plt.title('Segmented Image')
    plt.axis('off')
    
    print("\n7. Eroding an Image")
    
    kernel = np.ones((5, 5), np.uint8)
    eroded = cv2.erode(thresh_binary, kernel, iterations=1)
    
    dilated = cv2.dilate(thresh_binary, kernel, iterations=1)
    
    print("✓ Image erosion completed")
    
    plt.subplot(4, 5, 18)
    plt.imshow(eroded, cmap='gray')
    plt.title('Eroded Image')
    plt.axis('off')
    
    plt.subplot(4, 5, 19)
    plt.imshow(dilated, cmap='gray')
    plt.title('Dilated Image')
    plt.axis('off')
    
    print("\n8. Blurring an Image")
    
    blur_avg = cv2.blur(complex_img, (15, 15))
    blur_gaussian = cv2.GaussianBlur(complex_img, (15, 15), 0)
    blur_median = cv2.medianBlur(complex_img, 15)
    blur_bilateral = cv2.bilateralFilter(complex_img, 9, 75, 75)
    
    print("✓ Image blurring completed")
    
    plt.subplot(4, 5, 20)
    plt.imshow(cv2.cvtColor(blur_gaussian, cv2.COLOR_BGR2RGB))
    plt.title('Gaussian Blur')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    plt.figure(figsize=(15, 10))
    
    print("\n9. Create Border around Images")
    
    border_constant = cv2.copyMakeBorder(complex_img, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=[255, 0, 0])
    border_reflect = cv2.copyMakeBorder(complex_img, 20, 20, 20, 20, cv2.BORDER_REFLECT)
    border_replicate = cv2.copyMakeBorder(complex_img, 20, 20, 20, 20, cv2.BORDER_REPLICATE)
    border_wrap = cv2.copyMakeBorder(complex_img, 20, 20, 20, 20, cv2.BORDER_WRAP)
    
    print("✓ Border creation completed")
    
    plt.subplot(3, 4, 1)
    plt.imshow(cv2.cvtColor(complex_img, cv2.COLOR_BGR2RGB))
    plt.title('Original Image')
    plt.axis('off')
    
    plt.subplot(3, 4, 2)
    plt.imshow(cv2.cvtColor(border_constant, cv2.COLOR_BGR2RGB))
    plt.title('Constant Border')
    plt.axis('off')
    
    plt.subplot(3, 4, 3)
    plt.imshow(cv2.cvtColor(border_reflect, cv2.COLOR_BGR2RGB))
    plt.title('Reflect Border')
    plt.axis('off')
    
    plt.subplot(3, 4, 4)
    plt.imshow(cv2.cvtColor(border_replicate, cv2.COLOR_BGR2RGB))
    plt.title('Replicate Border')
    plt.axis('off')
    
    print("\n10. Edge Detection")
    
    edges_canny = cv2.Canny(complex_gray, 50, 150)
    
    sobel_x = cv2.Sobel(complex_gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(complex_gray, cv2.CV_64F, 0, 1, ksize=3)
    sobel_combined = np.sqrt(sobel_x**2 + sobel_y**2)
    sobel_combined = np.uint8(sobel_combined / sobel_combined.max() * 255)
    
    laplacian = cv2.Laplacian(complex_gray, cv2.CV_64F)
    laplacian = np.uint8(np.absolute(laplacian))
    
    print("✓ Edge detection completed")
    
    plt.subplot(3, 4, 5)
    plt.imshow(complex_gray, cmap='gray')
    plt.title('Original Grayscale')
    plt.axis('off')
    
    plt.subplot(3, 4, 6)
    plt.imshow(edges_canny, cmap='gray')
    plt.title('Canny Edges')
    plt.axis('off')
    
    plt.subplot(3, 4, 7)
    plt.imshow(sobel_combined, cmap='gray')
    plt.title('Sobel Edges')
    plt.axis('off')
    
    plt.subplot(3, 4, 8)
    plt.imshow(laplacian, cmap='gray')
    plt.title('Laplacian Edges')
    plt.axis('off')
    
    plt.subplot(3, 4, 9)
    plt.imshow(cv2.cvtColor(blur_avg, cv2.COLOR_BGR2RGB))
    plt.title('Average Blur')
    plt.axis('off')
    
    plt.subplot(3, 4, 10)
    plt.imshow(cv2.cvtColor(blur_median, cv2.COLOR_BGR2RGB))
    plt.title('Median Blur')
    plt.axis('off')
    
    plt.subplot(3, 4, 11)
    plt.imshow(cv2.cvtColor(blur_bilateral, cv2.COLOR_BGR2RGB))
    plt.title('Bilateral Filter')
    plt.axis('off')
    
    opening = cv2.morphologyEx(thresh_binary, cv2.MORPH_OPEN, kernel)
    closing = cv2.morphologyEx(thresh_binary, cv2.MORPH_CLOSE, kernel)
    
    plt.subplot(3, 4, 12)
    plt.imshow(opening, cmap='gray')
    plt.title('Opening')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    print("\n✓ All image processing operations completed!")
    print("\n=== Practice 2 Completed ===")
    
    print("\n=== Summary of Operations Performed ===")
    print("1. ✓ Arithmetic Operations: Addition, Subtraction, Blending")
    print("2. ✓ Bitwise Operations: AND, OR, XOR, NOT")
    print("3. ✓ Simple Thresholding: Binary, Binary Inverse, Truncate, To Zero")
    print("4. ✓ Adaptive Thresholding: Mean and Gaussian")
    print("5. ✓ Otsu Thresholding: Automatic threshold selection")
    print("6. ✓ Image Segmentation: Multi-level thresholding")
    print("7. ✓ Morphological Operations: Erosion, Dilation, Opening, Closing")
    print("8. ✓ Image Blurring: Average, Gaussian, Median, Bilateral")
    print("9. ✓ Border Creation: Constant, Reflect, Replicate, Wrap")
    print("10. ✓ Edge Detection: Canny, Sobel, Laplacian")

if __name__ == "__main__":
    practice_2() 