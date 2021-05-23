from scipy.ndimage import filters
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors


'''
Compute Harris operator for the image 
inputs: 
    image: Original gray image
    sigma: sigma of gaussian filter
output : Harris image
'''
'''
==================================================================
Put Your Code Here 
===================================================================
'''

def getHarris(image,sigma):

	global large,small
	#This part is written by the help of section 6 notes and  https://github.com/hughesj919/HarrisCorner/blob/master/Corners.py
	window_size = sigma
	dy, dx = np.gradient(image)
	Ixx = dx**2
	Ixy = dy*dx
	Iyy = dy**2
	height = image.shape[0]
	width = image.shape[1]
	offset = int(window_size/2)
	large = 0
	small = 0

	img = [[0 for x in range(width)] for y in range(height)]
	temp = [[0 for x in range(width)] for y in range(height)] 

	for y in range(0, height):
		for x in range(0, width):
			#Calculate sum of squares
			windowIxx = Ixx[y-offset:y+offset+1, x-offset:x+offset+1]
			windowIxy = Ixy[y-offset:y+offset+1, x-offset:x+offset+1]
			windowIyy = Iyy[y-offset:y+offset+1, x-offset:x+offset+1]
			Sxx = windowIxx.sum()
			Sxy = windowIxy.sum()
			Syy = windowIyy.sum()

			det = (Sxx * Syy) - (Sxy**2)
			trace = Sxx + Syy
			r = det - 0.2*(trace**2)

			img[y][x]=(r)
			if r > large:
				large = r
			if r<small:
				small = r

	H = img
	return H


def plotHarris(image, harrisIm, threshold):
	global large,small
	height = image.shape[0]
	width = image.shape[1]

	img = [[[0,0,0] for x in range(width)] for y in range(height)]

	for y in range(0, height):
		for x in range(0, width):
			if(harrisIm[y][x]*255 < (threshold*(large))) :
				img [y][x][0] = image[y][x][0]
				img [y][x][1] = image[y][x][1]
				img [y][x][2] = image[y][x][2]
			else:
				img[y][x] = [245,0,0]


	plt.figure("Image with corners")
	img = np.array(img)
	plt.imshow(img.astype(np.uint8))



if __name__ == '__main__':
    #Load image
    #Detect corners also in BW2.jpg which is similar to BW.jpg but with some rotation and
    #different illumination. What could you conclude from that?
    image = plt.imread('images/BW.jpg')
    #Extract value channel (intensity)
    hsvImage = colors.rgb_to_hsv(image)
    valIm = hsvImage[...,2]
    # Get Harris image
    harrIm = getHarris(valIm,3)
    #Show Original Image
    plt.figure("Original Image")
    plt.imshow(image)
    plt.set_cmap("gray")
    #show harris image
    plt.figure("Harris Image")
    plt.imshow(harrIm)
    # Show final image
    plotHarris(image, harrIm, 0.4)
    plt.show()






