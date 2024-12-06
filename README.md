
# 🖼️ Cartoonify: Transform Digital Images into Cartoon Art 🎨  

Cartoonify is a Django-based web application that transforms digital images into cartoon-style visuals using advanced image processing and machine learning techniques. The application provides an intuitive UI and seamless workflow, enabling users to upload their images, apply cartoon effects, and download the transformed output.  

---

## ✨ Features  
- **Cartoonize Images:** Convert your digital images into cartoon-style visuals effortlessly.  
- **Intuitive UI:** Easy-to-use interface with clean navigation.  
- **Real-Time Processing:** Upload, process, and download your cartoonized image within seconds.  
- **Advanced Techniques:** Uses sophisticated image processing techniques like edge detection, color quantization, and bilateral filtering.  

---

## 🛠️ Technologies Used  
- **Backend:** Django, Python  
- **Frontend:** HTML, CSS, JavaScript  
- **Libraries:** OpenCV, NumPy, Base64  

---

## ⚙️ How It Works  

The cartoonization process consists of the following steps:  

1. **Edge Detection**  
   Detects the edges in the image using adaptive thresholding.  

2. **Color Quantization**  
   Reduces the color palette using K-means clustering.  

3. **Bilateral Filtering**  
   Applies smoothing while retaining edges for a cartoon effect.   

4. **Final Cartoon Effect**  
   Combines the processed layers to produce the cartoonized image. 

---

## 🚀 Getting Started  

### Prerequisites  
- Python 3.8+  
- Django 4.0+  
- OpenCV, NumPy  

### Installation  
1. Clone the repository:  
   ```bash
   git clone https://github.com/deveshanurag/Artoon.git
   cd cartoonization
   ```  

2. Install dependencies:  
   ```bash
   pip install -r requirements.txt
   ```  

3. Run the server:  
   ```bash
   python manage.py runserver
   ```  

4. Open your browser and go to:  
   ```text
   http://127.0.0.1:8000/
   ```  

---

## 🖼️ Demo  
**Landing Page**
![Upload Page](https://github.com/deveshanurag/Artoon/blob/main/cartoonization/image/home1.png)  
![Upload Page](https://github.com/deveshanurag/Artoon/blob/main/cartoonization/image/home2.png)

---
**Choose one image and click cartoonize your image**


![Upload Page](https://github.com/deveshanurag/Artoon/blob/main/cartoonization/image/choosePic.png) 

---
**Sample like this**


![Upload Page](https://github.com/deveshanurag/Artoon/blob/main/cartoonization/image/sample.jpg)  


**Beautiful cartoonize image**


![Upload Page](https://github.com/deveshanurag/Artoon/blob/main/cartoonization/image/result.png)  

### Upload Image 


![Upload Page](https://github.com/deveshanurag/Artoon/blob/main/cartoonization/image/sample.jpg)  

### Cartoonized Image  
![Cartoonized Output](https://github.com/deveshanurag/Artoon/blob/main/cartoonization/image/final.jpeg)  

---

