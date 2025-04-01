import cv2
import numpy as np

def process_image(image_path):
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Could not load the image.")
        return
    
    # Perform an inverse transformation (negative image)
    inverted_image = cv2.bitwise_not(image)
    
    # Convert to grayscale for further processing
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Contrast stretching
    min_val, max_val, _, _ = cv2.minMaxLoc(gray_image)
    contrast_stretched = cv2.convertScaleAbs(gray_image, alpha=255.0/(max_val-min_val), beta=-min_val*255.0/(max_val-min_val))
    
    # Histogram equalization for contrast improvement
    histogram_equalized = cv2.equalizeHist(gray_image)
    
    # Edge detection using Canny edge detector
    edges = cv2.Canny(gray_image, 100, 200)

    # Save processed images
    cv2.imwrite("inverted.jpg", inverted_image)
    cv2.imwrite("contrast_stretched.jpg", contrast_stretched)
    cv2.imwrite("histogram_equalized.jpg", histogram_equalized)
    cv2.imwrite("edges.jpg", edges)

    print("Processed images have been saved successfully.")

# Example usage
image_path = "sample.jpg"  # Replace with your image path
process_image(image_path)