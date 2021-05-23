import numpy as np
import matplotlib.pyplot as plt
from skimage import feature
import time

def houghCircle(image):
    ''' Basic hough Circle transform that builds the accumulator array
    Input : image : edge image (canny)
    Output : accumulator : the accumulator of hough space (3D) space
    '''
    m, n = image.shape
    maxR = np.round((m**2 + n**2)**0.5)
    radInterval = [int(x) for x in np.arange(1, maxR)]
    accumulator = np.zeros((m, n, len(radInterval)))
    '''
    ==================================================================
    Put Your Code Here 
    ===================================================================
    '''

    R = 5
    for row in range(m):
        for col in range(n):
            if image[row, col] > 0:
                for theta in range(360):
                    for r in range(1, R):
                        a = int(round(row - r*np.cos(theta*np.pi / 180))) # polar coordinate for center
                        b = int(round(col - r*np.sin(theta*np.pi / 180)))  # polar coordinate for center
                        if a < m and b < n:
                            accumulator[a, b, r] += 1
    return accumulator


def detectCircles(image, accumulator, threshold):
    ''' Extract Circles with accumulator value > certain threshold
        Input : 
            image : Original image
            accumulator : Hough space (3D)
            threshold : fraction of max value in accumulator                
    '''
    '''
    ==================================================================
    Put Your Code Here 
    ===================================================================
    '''
    # Get maximum value in accumulator 
    # Now Sort accumulator to select top points
    # Initialzie lists of selected lines
    xind, yind, zind = np.nonzero(accumulator > threshold*np.amax(accumulator))
    detectedCircles = []
    for i in range(len(xind)):
        detectedCircles.append([xind[i], yind[i], zind[i]])
    # Now plot detected Circles in image
    plotCircles(image, detectedCircles)
        
def plotCircles(image, Circles):
    ''' Plot detected lines by detecLines method superimposed on original image
        input : image : original image
                lines : list of lines(r,theta)
    '''  
    '''
    ==================================================================
    Put Your Code Here 
    ===================================================================
    '''
    # Show image
    # Plot Circles overloaded on the image
    # just do nothing (Delete it after adding your code)
    plt.figure('Image after Hough transform')
    fig = plt.gcf()
    ax = fig.gca()
    plt.imshow(image)

    for circle in Circles:
        #plt.Circle((circle[0], circle[1]), circle[2], color='r')
        ax.add_artist(plt.Circle((circle[1], circle[0]), circle[2], color='r'))


if __name__ == '__main__':
    start_time = time.time()
    # Load the image
    image = plt.imread('images/coins.jpg')  
    # Edge detection (canny)
    edgeImage = feature.canny( image,sigma=1.4, low_threshold=40, high_threshold=150)    
    # Show original image
    plt.figure('Original Image')
    plt.imshow(image)
    plt.set_cmap('gray')
    # Show edge image
    plt.figure('Edge Image')
    plt.imshow(edgeImage)
    plt.set_cmap('gray')
    # build accumulator    
    accumulator = houghCircle(edgeImage)
    # Detect and superimpose lines on original image
    detectCircles(image, accumulator, 0.3)
    print("--- %0.3f seconds ---" % (time.time() - start_time))
    plt.show()