import cv2
import numpy as np

def read_img(filename):
    img = cv2.imread(filename)
    return img

def edge_detection(img, line_wdt, blur):
    # print(f"edge detection---->{img}")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    grayBlur = cv2.medianBlur(gray, blur)
    edges = cv2.adaptiveThreshold(grayBlur, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                  cv2.THRESH_BINARY, line_wdt, blur)
    return edges

def color_quantisation(img, k):
    data = np.float32(img).reshape((-1,3))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.001)
    ret, label, center = cv2.kmeans(data, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    center = np.uint8(center)
    result = center[label.flatten()]
    result = result.reshape(img.shape)
    return result

def cartoonify(filename, line_wdt=9, blur_value=7, totalColors=4):
    # Read the input image
    img = read_img(filename)
    
    # Apply edge detection
    edgeImg = edge_detection(img, line_wdt, blur_value)
    
    # Apply color quantization
    img_quantized = color_quantisation(img, totalColors)
    
    # Apply bilateral filter for smoothing
    blurred = cv2.bilateralFilter(img_quantized, d=7, sigmaColor=200, sigmaSpace=200)
    
    # Combine the edges and smoothed image using bitwise AND operation
    cartoon_img = cv2.bitwise_and(blurred, blurred, mask=edgeImg)
    
    return cartoon_img

# Example usage:

# input_image_path = './Kairav.jpg'
# cartoon_image = cartoonify(input_image_path)
# cv2.imwrite('cartoon_Kairav.jpg', cartoon_image)
