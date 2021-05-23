#Implement region growing segmentation. 
#Allow user to set an initial seed and then segment this region according to similarity of colors and or intensity.

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import timeit


# 1- Load color image
def load_image():
    image = cv2.imread("images/seg1.jpg")
    return image


def display_image(image):
    FIGURE_ROWS = 2
    FIGURE_COLS = 1
    ORIGINAL_INDEX = 1
    plt.subplot(FIGURE_ROWS, FIGURE_COLS, ORIGINAL_INDEX).imshow(image)
    plt.title('Original Image')

##################
# source https://stackoverflow.com/questions/43923648/region-growing-python?utm_medium=organic&utm_source=google_rich_qa&utm_campaign=google_rich_qa
##################
def calc_8_n(x, y, shape):
    out = []
    maxx = shape[0]-1
    maxy = shape[1]-1

    #top left
    outx = min(max(x-1,0),maxx)
    outy = min(max(y-1,0),maxy)
    out.append((outx,outy))

    #top center
    outx = x
    outy = min(max(y-1,0),maxy)
    out.append((outx,outy))

    #top right
    outx = min(max(x+1,0),maxx)
    outy = min(max(y-1,0),maxy)
    out.append((outx,outy))

    #left
    outx = min(max(x-1,0),maxx)
    outy = y
    out.append((outx,outy))

    #right
    outx = min(max(x+1,0),maxx)
    outy = y
    out.append((outx,outy))

    #bottom left
    outx = min(max(x-1,0),maxx)
    outy = min(max(y+1,0),maxy)
    out.append((outx,outy))

    #bottom center
    outx = x
    outy = min(max(y+1,0),maxy)
    out.append((outx,outy))

    #bottom right
    outx = min(max(x+1,0),maxx)
    outy = min(max(y+1,0),maxy)
    out.append((outx,outy))

    return out


def onclick(event):

    global x_coord, y_coord
    if event.xdata and event.ydata:
        x_coord = event.ydata
        y_coord = event.xdata
        seed = [int(x_coord), int(y_coord)]
        region_growing(seed, THRESHOLD=4)



def region_growing(seed, THRESHOLD):

    processed_pixels = np.zeros((image.shape[0], image.shape[1]))
    seeds = []
    FIGURE_ROWS = 2
    FIGURE_COLS = 1
    EDITED_INDEX = 2

    edited_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    segmented_image = np.full(processed_pixels.shape, 255, dtype=np.uint8)
    plt.subplot(FIGURE_ROWS, FIGURE_COLS, EDITED_INDEX).imshow(segmented_image, cmap=plt.get_cmap('gray'))
    plt.title('Segmented Image')
    plt.draw()


    seeds.append(seed)
    while len(seeds) > 0:
        seed = seeds.pop()
        edges = calc_8_n(seed[0], seed[1], edited_image.shape)
        for edge in edges:
            processed_pixels[seed[0], seed[1]] = 1
            if abs(int(edited_image[edge[0], edge[1]]) - int(edited_image[seed[0], seed[1]])) <= THRESHOLD and processed_pixels[edge[0], edge[1]] != 1:
                segmented_image[edge[0], edge[1]] = 0
                seeds.append(edge)
            else:
                #segmented_image[edge[0], edge[1]] = 255
                processed_pixels[edge[0], edge[1]] = 1
    plt.subplot(FIGURE_ROWS, FIGURE_COLS, EDITED_INDEX).imshow(segmented_image, cmap=plt.get_cmap('gray'))
    plt.title('Segmented Image')
    plt.draw()

    #edited_image = np.array(np.where(abs(edited_image - edited_image[seed[0], seed[1]]) <= THRESHOLD, 0, 255), dtype=np.uint8)





def main():
    global image
    image = load_image()

    # Create figure
    fig = plt.figure(4, figsize=(10, 6))
    fig.canvas.mpl_connect('button_press_event', onclick)
    fig.subplots_adjust(hspace=0.5, wspace=0.5)
    plt.suptitle("Region Growing")
    display_image(image)
    plt.show()


if __name__ == '__main__':
    main()
