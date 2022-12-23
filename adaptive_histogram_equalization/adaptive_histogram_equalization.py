import numpy as np
import cv2
import matplotlib.pyplot as plt

# Variable definitions

bin_Count       = 32
hist            = np.zeros(bin_Count)
hist_Red        = np.zeros(bin_Count)
hist_Green      = np.zeros(bin_Count)
hist_Blue       = np.zeros(bin_Count)
hist_RGB        = np.zeros(bin_Count*3)
winSize         = 129

def computeNormGrayHistogram(img):

    # Convert image to Grayscale
    img_Grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imshow('Grayscale image', img_Grayscale)
    # cv2.imwrite('Images_Q1/Grayscale_Image.png', img_Grayscale)
    cv2.waitKey(0)

    # Compute Histogram for Grayscale Image and Normalize it
    for i in range(bin_Count):
        hist[i] = ( ( ( img_Grayscale >= i*8 ) & ( img_Grayscale < (i+1)*8 ) ).sum() ) / ( np.size(img_Grayscale) )

    return hist

def computeNormRGBHistogram(img):

    # Function Variable Definitions
    img_Height                  = img.shape[0]
    img_Width                   = img.shape[1]
    pixels_Red_Intensity        = np.zeros((img_Height, img_Width))
    pixels_Green_Intensity      = np.zeros((img_Height, img_Width))
    pixels_Blue_Intensity       = np.zeros((img_Height, img_Width))

    # Convert image from OpenCV convention of BGR to RGB {For easy interpretation}
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Collect intensity values of each color
    for i in range(img_Height):
        for j in range(img_Width):
            pixels_Red_Intensity[i,j]       = img[i,j][0]
            pixels_Green_Intensity[i,j]     = img[i,j][1]
            pixels_Blue_Intensity[i,j]      = img[i,j][2]

    # Compute Histogram for each color and Normalize it
    for i in range(bin_Count):
        hist_RGB[i]         = ( ( ( pixels_Red_Intensity >= i*8 ) & ( pixels_Red_Intensity < (i+1)*8 ) ).sum() ) / (np.size(img))
        hist_RGB[32+i]      = ( ( ( pixels_Green_Intensity >= i*8 ) & ( pixels_Green_Intensity < (i+1)*8 ) ).sum() ) / (np.size(img))
        hist_RGB[64+i]      = ( ( ( pixels_Blue_Intensity >= i*8 ) & ( pixels_Blue_Intensity < (i+1)*8 ) ).sum() ) / (np.size(img))

    return hist_RGB

def modifyRChannel(img):

    # Function Variable Definitions
    img_Height                  = img.shape[0]
    img_Width                   = img.shape[1]
    pixels_Red_Intensity        = np.zeros((img_Height, img_Width))
    pixels_Green_Intensity      = np.zeros((img_Height, img_Width))
    pixels_Blue_Intensity       = np.zeros((img_Height, img_Width))

    # Convert image from OpenCV convention of BGR to RGB {For easy interpretation}
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Double the values of R Channel and collect intensity values of each color
    for i in range(img_Height):
        for j in range(img_Width):

            # Double the values of R Channel. If the intensity of red goes beyond 255 for any pixel, cap it at 255
            img[i,j][0]                     = 2 * img[i,j][0]
            if img[i,j][0] > 255:
                img[i,j][0] = 255

            # Collect the intensity values for each color
            pixels_Red_Intensity[i,j]       = img[i,j][0]
            pixels_Green_Intensity[i,j]     = img[i,j][1]
            pixels_Blue_Intensity[i,j]      = img[i,j][2]

    # Compute Histogram for each color and Normalize it
    for i in range(bin_Count):
        hist_RGB[i]         = ( ( ( pixels_Red_Intensity >= i*8 ) & ( pixels_Red_Intensity < (i+1)*8 ) ).sum() ) / (np.size(img))
        hist_RGB[32+i]      = ( ( ( pixels_Green_Intensity >= i*8 ) & ( pixels_Green_Intensity < (i+1)*8 ) ).sum() ) / (np.size(img))
        hist_RGB[64+i]      = ( ( ( pixels_Blue_Intensity >= i*8 ) & ( pixels_Blue_Intensity < (i+1)*8 ) ).sum() ) / (np.size(img))

    return img, hist_RGB

def adaptiveHistogramEqualization(img, winSize):

    # Function Variable Definitions
    img_Grayscale               = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_Height                  = img.shape[0]
    img_Width                   = img.shape[1]
    _enhanced_Image             = np.zeros((img_Height, img_Width))
    pads                        = int((winSize-1)/2)

    # Pad the image on all 4 sides using the OpenCV function
    padded_Image                = cv2.copyMakeBorder(img_Grayscale, pads, pads, pads, pads, cv2.BORDER_REFLECT)

    counter = 0
    for i in range(pads, img_Height + pads):
        for j in range(pads, img_Width + pads):
            rank = 0
            for x in range(-pads, pads + 1):
                for y in range(-pads, pads + 1):
                    if padded_Image[i,j] > padded_Image[i-x, j-y]:
                        rank = rank + 1
            _enhanced_Image[i-pads,j-pads] = (rank * 255)/(winSize*winSize)
        counter = counter + 1
        print(counter)

    return _enhanced_Image

def plotHistogramGrayscale(_hist, _bin_Count):

    x = np.arange(_bin_Count)

    plt.bar(x, _hist)
    plt.xlabel('Bins')
    plt.ylabel('Normalized value of no. of Pixels')
    plt.show()

    return 0

def plotHistogramRGB(_hist, _bin_Count):

    x = np.arange(_bin_Count)

    plt.bar(x[0:32], _hist[0:32], color = 'r')
    plt.bar(x[32:64], _hist[32:64], color = 'g')
    plt.bar(x[64:96], _hist[64:96], color = 'b')
    plt.xlabel('Bins')
    plt.ylabel('Normalized value of no. of Pixels')
    plt.show()

    return 0

# Read the image
img_1     = cv2.imread('forest.jpg')

# Plot Histogram for Grayscale Image
hist_Grayscale_Image    = computeNormGrayHistogram(img_1)
plt.title('Histogram of Grayscale Image')
plotHistogramGrayscale(hist_Grayscale_Image, bin_Count)

# Plot Histogram for RGB Image
hist_RGB_Image          = computeNormRGBHistogram(img_1)
plt.title('Histogram of RGB Image')
plotHistogramRGB(hist_RGB_Image, bin_Count*3)

# Plot histograms after flipping the image horizontally
img_2                   = cv2.flip(img_1, 1)
hist_RGB_Image          = computeNormRGBHistogram(img_2)
plt.title('Histogram of RGB Image after flipping horizontally')
plotHistogramRGB(hist_RGB_Image, bin_Count*3)

hist_Grayscale_Image    = computeNormGrayHistogram(img_2)
plt.title('Histogram of Grayscale Image after flipping horizontally')
plotHistogramGrayscale(hist_Grayscale_Image, bin_Count)

# Plot Histograms for Image with modified R Channel
modified_Img, hist_Modified_RGB_Image           = modifyRChannel(img_1)
cv2.imshow('Image with modified R Channel', modified_Img) # Show image with modified R Channel
# cv2.imwrite('Images_Q1/Image_Modified_R_Channel.png', modified_Img)
cv2.waitKey(0)
plt.title('Histogram of RGB Image with modified R Channel')
plotHistogramRGB(hist_Modified_RGB_Image, bin_Count*3)

hist_Grayscale_Image    = computeNormGrayHistogram(modified_Img)
plt.title('Histogram of Grayscale Image with modified R Channel')
plotHistogramGrayscale(hist_Grayscale_Image, bin_Count)

# Adaptive Histogram Equalization
img_new             = cv2.imread('beach.png')
enhanced_Image_AHE  = adaptiveHistogramEqualization(img_new, winSize)
cv2.imwrite('Images_Q1/Enhanced Image_129.png', enhanced_Image_AHE)

# Histogram Equalization
enhanced_Image_HE   = cv2.equalizeHist(cv2.cvtColor(img_new, cv2.COLOR_BGR2GRAY))
cv2.imwrite('Images_Q1/Enhanced Image_HE.png', enhanced_Image_HE)
