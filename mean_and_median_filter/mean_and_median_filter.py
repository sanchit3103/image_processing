import numpy as np
import cv2
import matplotlib.pyplot as plt

# Variable definitions

bin_Count       = 256
hist            = np.zeros(bin_Count)
winSize         = 5

def computeNormGrayHistogram(img):

    # Convert image to Grayscale
    img_Grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Compute Histogram for Grayscale Image and Normalize it
    for i in range(bin_Count):
        hist[i] = ( ( img_Grayscale == i ).sum() ) / ( np.size(img_Grayscale) )

    return hist

def meanFilter(img,winSize):

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
            _enhanced_Image[i-pads,j-pads] = np.mean( padded_Image[ i-pads:i+pads+1, j-pads:j+pads+1 ] )
        counter = counter + 1
        print(counter)

    return _enhanced_Image

def medianFilter(img,winSize):

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
            _enhanced_Image[i-pads,j-pads] = np.median( padded_Image[ i-pads:i+pads+1, j-pads:j+pads+1 ] )
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


# Read the image
img_mural               = cv2.imread('mural.jpg')
img_mural_noise_1       = cv2.imread('mural_noise1.jpg')
img_mural_noise_2       = cv2.imread('mural_noise2.jpg')
template                = cv2.imread('template.jpg')

# Plot Histogram for all 3 Mural Images
hist_mural              = computeNormGrayHistogram(img_mural)
plt.title('Histogram of Mural.jpg')
plotHistogramGrayscale(hist_mural, bin_Count)

hist_mural_noise_1      = computeNormGrayHistogram(img_mural_noise_1)
plt.title('Histogram of Mural_Noise1.jpg')
plotHistogramGrayscale(hist_mural_noise_1, bin_Count)

hist_mural_noise_2      = computeNormGrayHistogram(img_mural_noise_2)
plt.title('Histogram of Mural_Noise2.jpg')
plotHistogramGrayscale(hist_mural_noise_2, bin_Count)

# Enhancement using Mean Filter
enhanced_Img_mural_noise_1 = meanFilter(img_mural_noise_1, winSize)
cv2.imwrite('Images_Q2/Enhanced_Image_MeanF_81_Mural_Noise_1.png', enhanced_Img_mural_noise_1)

enhanced_Img_mural_noise_2 = meanFilter(img_mural_noise_2, winSize)
cv2.imwrite('Images_Q2/Enhanced_Image_MeanF_81_Mural_Noise_2.png', enhanced_Img_mural_noise_2)

# Plot Histogram of both the images after Mean Filter Enhancement
enhanced_Img_mural_noise_1 = cv2.imread('Images_Q2/Enhanced_Image_MeanF_81_Mural_Noise_1.png')
hist_mural_noise_1      = computeNormGrayHistogram(enhanced_Img_mural_noise_1)
plt.title('Histogram of Mural_Noise1 after Mean Filter, 81x81 neighborhood')
plotHistogramGrayscale(hist_mural_noise_1, bin_Count)

enhanced_Img_mural_noise_2 = cv2.imread('Images_Q2/Enhanced_Image_MeanF_81_Mural_Noise_2.png')
hist_mural_noise_2      = computeNormGrayHistogram(enhanced_Img_mural_noise_2)
plt.title('Histogram of Mural_Noise2 after Mean Filter, 81x81 neighborhood')
plotHistogramGrayscale(hist_mural_noise_2, bin_Count)

# Enhancement using Median Filter
enhanced_Img_mural_noise_1 = medianFilter(img_mural_noise_1, winSize)
cv2.imwrite('Images_Q2/Enhanced_Image_MedianF_81_Mural_Noise_1.png', enhanced_Img_mural_noise_1)

enhanced_Img_mural_noise_2 = medianFilter(img_mural_noise_2, winSize)
#cv2.imwrite('Images_Q2/Enhanced_Image_MedianF_81_Mural_Noise_2.png', enhanced_Img_mural_noise_2)

# Plot Histogram of both the images after Median Filter Enhancement
enhanced_Img_mural_noise_1 = cv2.imread('Images_Q2/Enhanced_Image_MedianF_81_Mural_Noise_1.png')
hist_mural_noise_1      = computeNormGrayHistogram(enhanced_Img_mural_noise_1)
plt.title('Histogram of Mural_Noise1 after Median Filter, 81x81 neighborhood')
plotHistogramGrayscale(hist_mural_noise_1, bin_Count)

enhanced_Img_mural_noise_2 = cv2.imread('Images_Q2/Enhanced_Image_MedianF_81_Mural_Noise_2.png')
hist_mural_noise_2      = computeNormGrayHistogram(enhanced_Img_mural_noise_2)
plt.title('Histogram of Mural_Noise2 after Median Filter, 81x81 neighborhood')
plotHistogramGrayscale(hist_mural_noise_2, bin_Count)

new_image = cv2.matchTemplate(img_mural, template, 'cv2.TM_CCORR')
cv2.imwrite('Images_Q2/After_Template.png', new_image)
