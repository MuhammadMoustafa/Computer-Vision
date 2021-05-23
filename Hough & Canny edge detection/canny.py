import numpy as np

from scipy import signal
from scipy.ndimage import filters
import matplotlib.pyplot as plt
from matplotlib import colors
from scipy import signal
from scipy import *
from scipy import ndimage



def doubleThresholding(image,tL,tH):
    low=np.where(image < tL)
    for x,y in zip (low[0],low[1]) :
        image[x, y] = 0
    high=np.where(image >tH)
    for x,y in zip (high[0],high[1]) :
        image[x, y] = np.max(image)
    return image

def hystresis(image):
    max=np.max(image)
    for x in range(1,np.shape(image)[0]-1):
        for y in range(1,np.shape(image)[1]-1):
           if(image[x+1,y]==max or image[x-1,y]==max or image[x,y+1]==max or image[x,y-1]==max ):
               # image[x,y]=max
             print (y)
           else:
               image[x,y]=0
        print(x)
    return image




def non_maximal_suppression(G, theta):
    """Performs non-maximal-suppression of gradients.
    Bins into 4 directions (up/down, left/right, both diagonals),
    and sets non-maximal elements in a 3x3 neighborhood to zero.
    Args:
        G: A (height, width) float numpy array of gradient magnitudes.
        theta: A (height, width) float numpy array of gradient directions.
    Returns:
        suppressed: A (height, width) float numpy array of suppressed
            gradient magnitudes.
    """

    theta *= 180.0 / np.pi
    theta[theta > 180.0] -= 180.0
    hits = np.zeros_like(G, dtype=bool)
    correlate = ndimage.correlate
    correlate1d = ndimage.correlate1d
    convolve = ndimage.convolve
    convolve1d = ndimage.convolve1d

    kernel = np.array([0.0, 1.0, -1.0])
    mask = np.logical_or(theta < 22.5, theta > 157.5)
    hits[mask] = np.logical_and(correlate1d(G, kernel, axis=-1)[mask] >= 0.0,
                                convolve1d(G, kernel, axis=-1)[mask] >= 0.0)

    mask = np.logical_and(theta >= 67.5, theta < 112.5)
    hits[mask] = np.logical_and(correlate1d(G, kernel, axis=0)[mask] >= 0.0,
                                convolve1d(G, kernel, axis=0)[mask] >= 0.0)

    kernel = np.array([[0.0, 0.0, 0.0],
                       [0.0, 1.0, 0.0],
                       [0.0, 0.0, -1.0]])
    mask = np.logical_and(theta >= 22.5, theta < 67.5)
    hits[mask] = np.logical_and(correlate(G, kernel)[mask] >= 0.0,
                                convolve(G, kernel)[mask] >= 0.0)

    kernel = np.array([[0.0, 0.0, 0.0],
                       [0.0, 1.0, 0.0],
                       [-1.0, 0.0, 0.0]])
    mask = np.logical_and(theta >= 112.5, theta < 157.5)
    hits[mask] = np.logical_and(correlate(G, kernel)[mask] >= 0.0,
                                convolve(G, kernel)[mask] >= 0.0)

    suppressed = G.copy()
    suppressed[np.logical_not(hits)] = 0.0

    return suppressed

def myCanny(image, tl, th):
    '''Canny edge detection algorithm 
    inputs : Grayscale image , tl : low threshold, th : high threshold
    output : Edge image or Canny image
    Basic steps of canny are : 
        1. Image smoothing using gaussian kernel for denoising
        2. Getting gradient magnitude image
        3. None maxima suppression: 
            Suppression of week edges at the same direction to have a thin edge
        4. Double thresholding : 
            Suppress globally weak edges that bellow tl, and keep that above th 
        5. Edge tracking:
            track remaining pixels with values in between tl and th. Suppress them
            if they haven't a strong edge in its neighbors.
    '''
    '''
    ==================================================================
    Put Your Code Here 
    ===================================================================
    '''

    sigma = 2
    gaussImage = ndimage.gaussian_filter(image, sigma)

    sobelY = np.array([[-1, -2, -1],
                       [0, 0, 0],
                       [1, 2, 1]])

    sobelX = np.array([[-1, 0, 1],
                       [-2, 0, 2],
                       [-1, 0, 1]])


    iX = signal.convolve2d(gaussImage, sobelX)

    iY = signal.convolve2d(gaussImage, sobelY)

    gradientMag = np.sqrt(iX ** 2 + iY ** 2)

    iTheta = np.arctan2(iY,iX)

    suppresedImage=non_maximal_suppression(gradientMag,iTheta)

    canny_image=suppresedImage


    canny_image = image

    canny_image = hystresis(doubleThresholding(suppresedImage,30,80)
                            )
    return canny_image



if __name__=='__main__':
    #Load Image
    image = plt.imread("images/Lines.jpg")
    #Extract value channel (intensity)
    hsvImage = colors.rgb_to_hsv(image)
    valIm = hsvImage[...,2]
    #Apply canny on the image
    cannyIm = myCanny(valIm, 50, 100)
    #Show Original Image
    plt.figure("Original Image")
    plt.imshow(valIm)
    plt.set_cmap("gray")
    #Show Canny image
    plt.figure("Canny Image")
    plt.imshow(cannyIm)
    plt.show()
            
    
        
        
                
        
    

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
