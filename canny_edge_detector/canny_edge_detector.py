import numpy as np
import cv2
import matplotlib.pyplot as plt

# Variable definitions

bin_Count       = 256
hist            = np.zeros(bin_Count)
k               = (1/159) * np.array([[2, 4, 5, 4, 2],
                                      [4, 9, 12, 9, 4],
                                      [5, 12, 15, 12, 5],
                                      [4, 9, 12, 9, 4],
                                      [2, 4, 5, 4, 2]])

kx              = np.array([[-1, 0, 1],
                            [-2, 0, 2],
                            [-1, 0, 1]])

ky              = np.array([[-1, -2, -1],
                            [0, 0, 0],
                            [1, 2, 1]])


def smoothing(img):

    # Function Variable Definitions
    img_Height                  = img.shape[0]
    img_Width                   = img.shape[1]
    _enhanced_Image             = np.zeros((img_Height, img_Width))
    pads                        = int((k.shape[0]-1)/2)

    # Pad the image on all 4 sides using the OpenCV function
    padded_Image                = cv2.copyMakeBorder(img, pads, pads, pads, pads, cv2.BORDER_REFLECT)

    for i in range(pads, img_Height + pads):
        for j in range(pads, img_Width + pads):
            _enhanced_Image[i-pads,j-pads] = np.sum( padded_Image[ i-pads:i+pads+1, j-pads:j+pads+1 ] * k )

    return _enhanced_Image

def gradient(img):

    # Function Variable Definitions
    img_Height                  = img.shape[0]
    img_Width                   = img.shape[1]
    gradient_Image_x            = np.zeros((img_Height, img_Width))
    gradient_Image_y            = np.zeros((img_Height, img_Width))
    _gradient_Mag_Img           = np.zeros((img_Height, img_Width))
    _edge_Dir_Img               = np.zeros((img_Height, img_Width))
    pads                        = int((kx.shape[0]-1)/2)

    # Pad the image on all 4 sides using the OpenCV function
    padded_Image                = cv2.copyMakeBorder(img, pads, pads, pads, pads, cv2.BORDER_REFLECT)

    for i in range(pads, img_Height + pads):
        for j in range(pads, img_Width + pads):
            gradient_Image_x[i-pads,j-pads] = np.sum( padded_Image[ i-pads:i+pads+1, j-pads:j+pads+1 ] * kx )
            gradient_Image_y[i-pads,j-pads] = np.sum( padded_Image[ i-pads:i+pads+1, j-pads:j+pads+1 ] * ky )

    for i in range(img_Height):
        for j in range(img_Width):
            _gradient_Mag_Img[i,j]      = int( np.sqrt( gradient_Image_x[i,j] ** 2 + gradient_Image_y[i,j] ** 2 ) )
            if _gradient_Mag_Img[i,j] > 255:
                _gradient_Mag_Img[i,j]  = 255
            _edge_Dir_Img[i,j]          = int( np.degrees( np.arctan( gradient_Image_y[i,j]/gradient_Image_x[i,j] ) ) )
            if _edge_Dir_Img[i,j] > 255:
                _edge_Dir_Img[i,j]      = 255

    return _gradient_Mag_Img, _edge_Dir_Img

def nms(_gradient_Mag_Img, _edge_Dir_Img):

    # Function Variable Definitions
    img_Height                  = _gradient_Mag_Img.shape[0]
    img_Width                   = _gradient_Mag_Img.shape[1]
    _img_nms_step               = np.zeros((img_Height, img_Width), dtype=np.int32)
    pads                        = 1
    pixel_value_1               = 0
    pixel_value_2               = 0

    # Pad the Gradient Magnitude image on all 4 sides using the OpenCV function
    padded_Image                = cv2.copyMakeBorder(_gradient_Mag_Img, pads, pads, pads, pads, cv2.BORDER_REFLECT)

    for i in range(pads, img_Height + pads):
        for j in range(pads, img_Width + pads):
            # Round to 0 degrees
            if ( 0 <= _edge_Dir_Img[i-pads,j-pads] < 22.5) or ( -22.5 <= _edge_Dir_Img[i-pads,j-pads] <= 0):
                pixel_value_1 = padded_Image[i,j+1]
                pixel_value_2 = padded_Image[i,j-1]

            # Round to 45 degrees
            elif ( 22.5 <= _edge_Dir_Img[i-pads,j-pads] < 67.5):
                pixel_value_1 = padded_Image[i-1,j+1]
                pixel_value_2 = padded_Image[i+1,j-1]

            # Round to -45 degrees
            elif ( -67.5 <= _edge_Dir_Img[i-pads,j-pads] < -22.5):
                pixel_value_1 = padded_Image[i-1,j-1]
                pixel_value_2 = padded_Image[i+1,j+1]

            elif ( 67.5 <= _edge_Dir_Img[i-pads,j-pads] < 90) or ( -90 <= _edge_Dir_Img[i-pads,j-pads] <= -67.5):
                pixel_value_1 = padded_Image[i-1,j]
                pixel_value_2 = padded_Image[i+1,j]

            if (padded_Image[i,j] >= pixel_value_1) and (padded_Image[i,j] >= pixel_value_2):
                _img_nms_step[i-pads,j-pads] = padded_Image[i,j]

    return _img_nms_step

def thresholding(_nms_img):

    # Function Variable Definitions
    ratio_High_Threshold    = 0.20
    ratio_Low_Threshold     = 0.08
    high_Threshold          = _nms_img.max() * ratio_High_Threshold
    low_Threshold           = high_Threshold * ratio_Low_Threshold
    weak_Pixel_Value        = 20
    strong_Pixel_Value      = 255
    img_Height              = _nms_img.shape[0]
    img_Width               = _nms_img.shape[1]
    _thresholding_img       = np.zeros((img_Height, img_Width), dtype=np.int32)

    strong_Pixel_i, strong_Pixel_j  = np.where(_nms_img >= high_Threshold)
    weak_Pixel_i, weak_Pixel_j      = np.where( (_nms_img < high_Threshold) & (_nms_img >= low_Threshold))

    _thresholding_img[strong_Pixel_i, strong_Pixel_j]   = strong_Pixel_Value
    _thresholding_img[weak_Pixel_i, weak_Pixel_j]       = weak_Pixel_Value

    return _thresholding_img

# Read the image
input_img               = cv2.imread('lane.png')

# Convert the image to Grayscale
img_Grayscale           = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Smoothen the Image by applying the Gaussian Filter
smoothed_Img            = smoothing(img_Grayscale)
cv2.imwrite('Images_Q3/Smoothed_Image.png', smoothed_Img)

# Compute Gradient Magnitude Image
gradient_Magnitude_Img, edge_Direction_Img  = gradient(smoothed_Img)
np.savetxt('edge_Direction_Img.txt', edge_Direction_Img, fmt = "%s")
cv2.imwrite('Images_Q3/Gradient_Magnitude_Image.png', gradient_Magnitude_Img)
cv2.imwrite('Images_Q3/Edge_Direction_Image.png', edge_Direction_Img)

# Image after Non-maximum Suppression (NMS)
img_NMS_Step            = nms(gradient_Magnitude_Img, edge_Direction_Img)
cv2.imwrite('Images_Q3/Image_NMS_Step.png', img_NMS_Step)

img2 = thresholding(img_Grayscale)
cv2.imwrite('Images_Q3/Thresholding.png', img2)
