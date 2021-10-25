import cv2
import numpy as np
from PIL import Image

def change_range(data, input_min, input_max, output_min, output_max, eps=1e-8):
    result = ((data - input_min) / (input_max - input_min+eps)) * (output_max - output_min) + output_min
    return result

def read_grayscale_img(path):
    img = Image.open(path).convert('L')
    m, n = img.getdata().size
    img = np.asarray(img.getdata()).reshape(n, m, 1)
    return img

def clahe_equalized(img):
    assert (len(img.shape) == 2)  # 2D image
    assert issubclass(img.dtype.type, np.uint8) # unsigned int 8 bits
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)) #create a CLAHE object
    img_equalized = clahe.apply(img)
    return img_equalized

def adjust_gamma(img, gamma=1.0):
    assert issubclass(img.dtype.type, np.uint8)  # unsigned int 8 bits
    # build a lookup table mapping the pixel values [0, 255] to their adjusted gamma values
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    # apply gamma correction using the lookup table
    new_img = cv2.LUT(img, table)
    return new_img


