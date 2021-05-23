import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.colors as color
from datetime import datetime
from scipy import signal
from scipy.ndimage import filters


def fastDetect(image, t):
    '''
    Check if point is a corner using 16 point on the circle. It will be a corner
    if you have 5 or more pixels with absolute differenc from center pixel that 
    greater than threshold t. 
    
    Original algorithm is more sophisticated. Advanced machine learing is applied.
    Here you are asked to implement the basic idea with no more improvements. 
    Also we will ignore non-maxima suppression step. 
    
    inputs : 
          image : gray image
          t : threshold 
    output : Corner image
    '''
    '''
    ==================================================================
    Put Your Code Here 
    ===================================================================
    '''
    # axis 0 for rows +ve down ....... axis 1 for cols +ve right

    image1 = np.roll(image, -3, axis=0)
    image2 = np.roll(np.roll(image, -3, axis=0), 1, axis=1)
    image3 = np.roll(np.roll(image, -2, axis=0), 2, axis=1)
    image4 = np.roll(np.roll(image, -1, axis=0), 3, axis=1)
    image5 = np.roll(image, 3, axis=1)
    image6 = np.roll(np.roll(image, 1, axis=0), 3, axis=1)
    image7 = np.roll(np.roll(image, 2, axis=0), 2, axis=1)
    image8 = np.roll(np.roll(image, 3, axis=0), 1, axis=1)
    image9 = np.roll(image, 3, axis=0)
    image10 = np.roll(np.roll(image, 3, axis=0), -1, axis=1)
    image11 = np.roll(np.roll(image, 2, axis=0), -2, axis=1)
    image12 = np.roll(np.roll(image, 1, axis=0), -3, axis=1)
    image13 = np.roll(image, -3, axis=1)
    image14 = np.roll(np.roll(image, -1, axis=0), -3, axis=1)
    image15 = np.roll(np.roll(image, -2, axis=0), -2, axis=1)
    image16 = np.roll(np.roll(image, -3, axis=0), -1, axis=1)

    cornerImage1 = 1 * (abs((image - image1)) >= t)
    cornerImage2 = 1 * (abs((image - image2)) >= t)
    cornerImage3 = 1 * (abs((image - image3)) >= t)
    cornerImage4 = 1 * (abs((image - image4)) >= t)
    cornerImage5 = 1 * (abs((image - image5)) >= t)
    cornerImage6 = 1 * (abs((image - image6)) >= t)
    cornerImage7 = 1 * (abs((image - image7)) >= t)
    cornerImage8 = 1 * (abs((image - image8)) >= t)
    cornerImage9 = 1 * (abs((image - image9)) >= t)
    cornerImage10 = 1 * (abs((image - image10)) >= t)
    cornerImage11 = 1 * (abs((image - image11)) >= t)
    cornerImage12 = 1 * (abs((image - image12)) >= t)
    cornerImage13 = 1 * (abs((image - image13)) >= t)
    cornerImage14 = 1 * (abs((image - image14)) >= t)
    cornerImage15 = 1 * (abs((image - image15)) >= t)
    cornerImage16 = 1 * (abs((image - image16)) >= t)

    cornerImage = cornerImage1 + cornerImage2 + cornerImage3 + cornerImage4 + cornerImage5 + cornerImage6 \
        + cornerImage7 + cornerImage8 + cornerImage9 + cornerImage10 + cornerImage11 + cornerImage12 + cornerImage13 \
        + cornerImage14 + cornerImage15 + cornerImage16
    cornerImage[cornerImage < 5] = 0
    cornerImage[cornerImage >= 5] = 255

    return cornerImage


def plotCorners(image, cornerImage):
    '''
    Plot detected corners from corner image superimposed on original image
    '''
    '''
    ==================================================================
    Put Your Code Here 
    ===================================================================
    '''
    #Show original image 
    # Mark up the corners on it
    plt.figure("Corner Image after Fast")
    plt.imshow(image)

    for row in range(cornerImage.shape[0]):
        for col in range(cornerImage.shape[1]):
            if cornerImage[row, col] > 0:
                #plt.plot(col, row, 'rx')
                # Courtesy of Mohammed Khalaf
                image.flags.writeable = True
                image[row, col, :] = [255, 0, 0]
    plt.imshow(image)


if __name__ == '__main__':
    startTime = datetime.now()
    #Load Image
    image = plt.imread("images/BW.jpg")
    hsvImage = color.rgb_to_hsv(image)
    valIm = hsvImage[..., 2]
    # Detect corners
    cornerImage = fastDetect(valIm, 75)
    #Show Original Image
    plt.figure("Original Image")
    plt.imshow(image)
    plt.set_cmap("gray")
    #show corner image
    plt.figure("Corner Image")
    plt.imshow(cornerImage)
    #Plot corners
    plotCorners(image, cornerImage)
    print(datetime.now() - startTime)
    plt.show()