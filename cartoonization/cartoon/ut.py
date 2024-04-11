import cv2
import numpy as np
import base64
from django.http import HttpResponse
def convert_to_gray(img):
    # Convert the image to grayscale manually
    gray = np.dot(img[..., :3], [0.2989, 0.5870, 0.1140])
    return gray

def median_blur(img, ksize):
    # Apply median blur manually
    rows, cols = img.shape
    blurred = np.zeros_like(img)
    for i in range(1, rows-1):
        for j in range(1, cols-1):
            neighbors = [img[i-1, j-1], img[i-1, j], img[i-1, j+1],
                         img[i, j-1], img[i, j], img[i, j+1],
                         img[i+1, j-1], img[i+1, j], img[i+1, j+1]]
            blurred[i, j] = sorted(neighbors)[4]  # Pick the median value
    return blurred

def adaptive_threshold(img, max_val, method, block_size, C):
    # Apply adaptive thresholding manually
    rows, cols = img.shape
    thresholded = np.zeros_like(img)
    for i in range(1, rows-1):
        for j in range(1, cols-1):
            block = img[i-block_size//2:i+block_size//2+1, j-block_size//2:j+block_size//2+1]
            block_mean = np.mean(block)
            if img[i, j] > block_mean - C:
                thresholded[i, j] = max_val
    return thresholded

def edge_detection(img, line_wdt, blur):
    # Convert to grayscale
    gray = convert_to_gray(img)
    
    # Apply median blur
    gray_blur = median_blur(gray, blur)
    
    # Apply adaptive thresholding
    edges = adaptive_threshold(gray_blur, 255, "mean", line_wdt, 0)
    
    return edges

def kmeans_clustering(img, k, max_iter=10):
    # Randomly initialize centroids
    centroids = img[np.random.choice(img.shape[0], k, replace=False)]
    
    for _ in range(max_iter):
        # Assign each pixel to the nearest centroid
        distances = np.sqrt(np.sum((img - centroids[:, np.newaxis])**2, axis=2))
        labels = np.argmin(distances, axis=0)
        
        # Update centroids
        for i in range(k):
            centroids[i] = np.mean(img[labels == i], axis=0)
    
    return centroids, labels

def color_quantisation(img, k):
    # Flatten the image to 1D array
    flattened_img = img.reshape(-1, 3)
    
    # Run k-means clustering
    centroids, labels = kmeans_clustering(flattened_img, k)
    
    # Assign each pixel to the nearest centroid
    quantized_img = np.zeros_like(flattened_img)
    for i in range(k):
        quantized_img[labels == i] = centroids[i]
    
    # Reshape the quantized image back to original shape
    result = quantized_img.reshape(img.shape)
    
    return result
def bilateral_filter(img, d, sigma_color, sigma_space):
    # Define the kernel size
    kernel_size = 2 * d + 1
    
    # Initialize the output image
    filtered_img = np.zeros_like(img)
    
    # Iterate over each pixel in the image
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            # Define the region of interest
            roi = img[max(0, i - d): min(img.shape[0], i + d + 1),
                      max(0, j - d): min(img.shape[1], j + d + 1)]
            
            # Calculate the Gaussian kernel weights
            color_weights = np.exp(-((roi - img[i, j])**2) / (2 * sigma_color**2))
            space_weights = np.exp(-((np.arange(roi.shape[0]) - i)**2 +
                                     (np.arange(roi.shape[1]) - j)**2) / (2 * sigma_space**2))
            
            # Apply the bilateral filter
            filtered_img[i, j] = np.sum(roi * color_weights * space_weights) / np.sum(color_weights * space_weights)
    
    return filtered_img

def bitwise_and(img1, img2, mask):
    # Initialize the output image
    result_img = np.zeros_like(img1)
    
    # Apply the bitwise AND operation
    result_img[mask != 0] = img1[mask != 0]
    
    return result_img



def cartoonize_image(image):
    # Read image using OpenCV
    img = cv2.imdecode(np.frombuffer(image.read(), np.uint8), cv2.IMREAD_COLOR)
    print(img)
    
    # Convert image to grayscale
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = edge_detection(img,9,7)

    img_quantized = color_quantisation(img, 4)
    blurred = cv2.bilateralFilter(img_quantized, 7, 200, 200)
    
    
    # Apply bilateral filter to reduce noise while keeping edges sharp
    # gray = cv2.bilateralFilter(gray, 9, 75, 75)
    
    # Apply median blur to further reduce noise
    # gray = cv2.medianBlur(gray, 7)
    
    # Detect edges using Canny edge detector
    # edges = cv2.Canny(gray, 100, 200)
    
    # Create a cartoon-like effect by combining edges with a color image
    # color = cv2.bilateralFilter(img, 7, 200, 200)
    cartoon = cv2.bitwise_and(blurred, blurred, mask=gray)
    
    # Convert cartoon image to base64 string for displaying in HTML
    _, img_encoded = cv2.imencode('.jpg', cartoon)
    cartoon_base64 = base64.b64encode(img_encoded).decode('utf-8')
    
    # return cartoon_base64

    response = HttpResponse(img_encoded.tobytes(), content_type='image/jpeg')
    
    # Set Content-Disposition header to force download
    response['Content-Disposition'] = 'attachment; filename="cartoon_image.jpg"'
    
    return cartoon_base64, response
