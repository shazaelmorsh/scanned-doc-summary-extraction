import cv2
import numpy as np
import matplotlib.pyplot as plt

def display_from_memory(image):
    plt.figure(figsize=(8, 6))
    plt.imshow(image, cmap='gray')
    plt.axis('off')
    plt.show()
    
def display_from_path(im_path):
    dpi = 80
    im_data = plt.imread(im_path)
    height, width = im_data.shape[:2]
    figsize = width / float(dpi), height / float(dpi)
    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis('off')
    ax.imshow(im_data, cmap='gray')

    plt.show()

def binarization(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh, binarized_img = cv2.threshold(gray_image, 210, 230, cv2.THRESH_BINARY)
    return binarized_img

def noise_removal(image):
    kernel = np.ones((1, 1), np.uint8)
    image = cv2.dilate(image, kernel, iterations=1)
    kernel = np.ones((1, 1), np.uint8)
    image = cv2.erode(image, kernel, iterations=1)
    image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    image = cv2.medianBlur(image, 3)
    return image

def thin_font(image):
    image = cv2.bitwise_not(image)
    kernel = np.zeros((2,2),np.uint8)
    image = cv2.erode(image, kernel, iterations=1)
    image = cv2.bitwise_not(image)
    return image

def thick_font(image):
    image = cv2.bitwise_not(image)
    kernel = np.ones((2,2),np.uint8)
    image = cv2.dilate(image, kernel, iterations=1)
    image = cv2.bitwise_not(image)
    return image

def remove_borders(image):
    contours, hierarchy = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cntsSorted = sorted(contours, key=lambda x:cv2.contourArea(x))
    cnt = cntsSorted[-1]
    x, y, w, h = cv2.boundingRect(cnt)
    crop = image[y:y+h, x:x+w]
    return crop

def preprocess_pipeline(image_path):
    # Load the image
    img = cv2.imread(image_path)
    
    # #for debugging
    # display_from_path(image_path)
    
    # Change to binary
    gray_image = binarization(img)
    
    # Noise removal
    no_noise = noise_removal(gray_image)
    
    # Thin font processing
    # eroded_image = thin_font(no_noise)
    
    # Thick font processing
    dilated_image = thick_font(no_noise)
    
    # Remove borders
    # no_borders = remove_borders(dilated_image)
    
    #for debugging
    # display_from_memory(dilated_image)
    
    return dilated_image
