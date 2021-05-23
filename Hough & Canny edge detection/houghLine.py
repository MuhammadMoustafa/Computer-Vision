import numpy as np
import matplotlib.pyplot as plt
from skimage import feature
import matplotlib.colors as color
import time


def houghLine(image):
    ''' Basic hough line transform that builds the accumulator array
    Input : image : edge image (canny)
    Output : accumulator : the accumulator of hough space
             thetas : values of theta (-90 : 90)
             rs : values of radius (-max distance : max distance)
    '''
    # Theta in range from -90 to 90 degrees
    thetas = np.deg2rad(np.arange(-90, 90)) 
    #Get image dimensions
    # y for rows and x for columns 
    Ny = image.shape[0]
    Nx = image.shape[1] 
    #Max diatance is diagonal one 
    Maxdist = int(np.round(np.sqrt(Nx**2 + Ny ** 2))) 
    #Range of radius
    rs = np.linspace(-Maxdist, Maxdist, 2*Maxdist) 
    # initialize accumulator array to zeros
    accumulator = np.zeros((2 * Maxdist, len(thetas))) 
    # Now Start Accumulation
    '''
    ==================================================================
    Put Your Code Here 
    ===================================================================
    '''

    for y in range(Ny):
        for x in range(Nx):
            # Check if it is an edge pixel
            #  NB: y -> rows , x -> columns
            if image[y, x] > 0:
                for k in range(len(thetas)):
                    r = x*np.cos(thetas[k]) + y * np.sin(thetas[k])
                    accumulator[int(r) + Maxdist, k] += 1

    return accumulator, thetas, rs


def detectLines(image,accumulator, threshold, rohs, thetas):
    ''' Extract lines with accumulator value > certain threshold
        Input : image : Original image
                accumulator: Hough space
                threshold : fraction of max value in accumulator
                rhos : radii array ( -dmax : dmax)
                thetas : theta array
                
    '''
    '''
    ==================================================================
    Put Your Code Here 
    ===================================================================
    '''

    # Get maximum value in accumulator
    # Now Sort accumulator to select top points
    # Initialize lists of selected lines


    detectedLinesInd = np.argwhere(accumulator > threshold*np.amax(accumulator))

    detectedLines = []
    for i in range(len(detectedLinesInd)):
        detectedLines.append([detectedLinesInd[i, 0] - len(rohs) / 2, thetas[detectedLinesInd[i, 1]]])

    # another way ---> c = map(lambda x,y:(x,y),a,b)

    # Check current value relative to threshold value
    # Add line if value > threshold
    # Now plot detected lines in image
    plotLines(image, detectedLines)


def plotLines(image, lines):
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
    # Plot lines overloaded on the image
    # just do nothing (Delete it after adding your code)
    plt.figure('Image after Hough transform')
    plt.imshow(image)

    for line in lines:

        x = np.arange(0, image.shape[1])
        y = (np.full(len(x), line[0]/np.sin(line[1])) - x*np.cos(line[1])/np.sin(line[1]))
        y = y.clip(min=0, max=image.shape[0])
        plt.plot(x, y, '-b')



if __name__ == '__main__':

    '''
    #timing -> Courtesy of omarcartera
    '''

    start_time = time.time()
    # Load the image
    image = plt.imread('images/Lines.jpg')
    # Get value Channel (intensity)
    hsvImage = color.rgb_to_hsv(image)
    ValImage = hsvImage[..., 2]
    # Detect edges using canny 
    edgeImage = feature.canny(ValImage, sigma=1.4, low_threshold=40, high_threshold=150)
    # Show original image
    plt.figure('Original Image')
    plt.imshow(image)
    plt.set_cmap('gray')
    # Show edge image
    plt.figure('Edge Image')
    plt.imshow(edgeImage)
    plt.set_cmap('gray')
    # build accumulator    
    accumulator, thetas, rhos = houghLine(edgeImage)
    # Visualize hough space
    plt.figure('Hough Space')
    plt.imshow(accumulator)
    plt.set_cmap('gray')
    # Detect and superimpose lines on original image
    detectLines(image, accumulator, 0.3, rhos, thetas)
    print("--- %0.3f seconds ---" % (time.time() - start_time))
    plt.show()
