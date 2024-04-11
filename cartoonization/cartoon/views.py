from django.shortcuts import render
from django.http import HttpResponse
from .utilis import cartoonize_image
# from .ut import cartoonize_image
# from .model import cartoonify
import cv2
import numpy as np


def index(request):
    return render(request, 'cartoon/index.html')

def result(request):
    if request.method == 'POST' and request.FILES['image']:
        image = request.FILES['image']
        # cartoon_image = cartoonize_image(image)
        cartoon_image, download_response = cartoonize_image(image)
        return render(request, 'cartoon/result.html', {'cartoon_image': cartoon_image, 'download_response': download_response})
    return render(request, 'cartoon/result.html')

    #     return render(request, 'cartoon/result.html', {'cartoon_image': cartoon_image})
    # return render(request, 'cartoon/result.html')
    

def test(request):
    if request.method == 'POST' and request.FILES['image']:
        image = request.FILES['image']
        print(image)
        cartoon_image = cartoonize_image(image)
        return render(request, 'cartoon/test.html', {'cartoon_image': cartoon_image})
    return render(request, 'cartoon/test.html')
