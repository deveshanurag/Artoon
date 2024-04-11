import cv2
import numpy as np
import base64
from django.http import HttpResponse
def edge_detection(img, line_wdt, blur):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    grayBlur = cv2.medianBlur(gray, blur)
    # print(f"edge detection----->{grayBlur}")
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
    # print(f"color-quantization done")
    return result

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
