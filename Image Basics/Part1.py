import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import timeit


# 1- Load color image
def load_images(image_paths):
    colored_images = [mpimg.imread(image_path) for image_path in image_paths]
    gray_images = [cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) for image in colored_images]

    return colored_images, gray_images


# 2- Display RGB on a screen
def display_images(images):

    fig = plt.figure(4, figsize=(10, 6))
    fig.subplots_adjust(hspace=0.5, wspace=0.5)
    plt.suptitle("Load Images")

    figure_cols = 2
    figure_rows = len(images)
    x_range = np.arange(256)

    index = 0
    for i in range(len(images)):
        index += 1
        plt.subplot(figure_rows, figure_cols, index).imshow(images[i])

        index += 1
        red_hist, green_hist, blue_hist = display_histogram(images[i])
        plt.subplot(figure_rows, figure_cols, index).plot(x_range, red_hist, 'r', x_range, green_hist, 'g', x_range, blue_hist, 'b')



# 3- Display the histograms of all three colour channels, for each Image, on the same figure
def display_histogram(image):

    red_hist, _ = np.histogram(image[:, :, 0].flatten(), bins=256)
    green_hist, _ = np.histogram(image[:, :, 1].flatten(), bins=256)
    blue_hist, _ = np.histogram(image[:, :, 2].flatten(), bins=256)

    return red_hist, green_hist, blue_hist

# 4- Select the image of our visitor of this week, Sir Pegion. Move the mouse cursor within your image.
# For the current pixel location p in the image, compute and display

def image_stat_show(image, location=None):
    fig = plt.figure(3, figsize=(10, 6))
    fig.canvas.mpl_connect('button_press_event', onclick)
    if location:

        image_x = int(np.rint(x_coord))
        image_y = int(np.rint(y_coord))

        image_window = sir_pigeon[image_x - 5:image_x + 6, image_y - 5:image_y + 6, :]
        rgb = sir_pigeon[image_x, image_y, :]
        intensity = np.rint(np.mean(rgb))
        window_mean = np.rint(np.mean(image_window))
        window_std = np.rint(np.std(image_window))

        plt.title('location: row = {} , col = {},  r = {}, g = {}, b = {} '.format(np.rint(location[0]),
                                                                                      np.rint(location[1]), rgb[0], rgb[1], rgb[2]))
        plt.xlabel('Intensity = {}, mean = {}, std = {}, variance = {}'.format(intensity, window_mean, window_std,
                                                                               window_std**2))

    plt.imshow(image)
    plt.draw()


def image_stat():
    global x_coord
    global y_coord
    global sir_pigeon

    pixel_location = [x_coord, y_coord]
    image_stat_show(sir_pigeon, location=pixel_location)


def onclick(event):

    global x_coord
    global y_coord

    if event.xdata and event.ydata:
        x_coord = event.ydata
        y_coord = event.xdata
        image_stat()


# 5- Compute the gradient of the image
# Implement a version without using explicit for loops.
def image_gradient_vectorized(image, brightness=1):

    image_x = np.roll(image, 1, 0) - image
    image_y = np.roll(image, 1, 1) - image

    gradient_image = ((image_x ** 2 + image_y ** 2) ** 0.5) * brightness
    gradient_image = np.asanyarray(gradient_image, dtype=image.dtype)

    return gradient_image


# 5- Compute the gradient of the image
# Implement a version with explicit for loops.
def image_gradient_loop(image, brightness=1):

    gradient_image = np.zeros(image.shape)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            x_gradient = image[i, j] - image[i, j-1]
            y_gradient = image[i, j] - image[i-1, j]
            gradient_image[i, j] = ((x_gradient**2 + y_gradient**2)**0.5) * brightness

    gradient_image = np.asanyarray(gradient_image, image.dtype)
    return gradient_image


def display_image_gradient(colored_images):
    fig = plt.figure(2, figsize=(10, 6))
    fig.subplots_adjust(hspace=0.5, wspace=0.5)
    plt.suptitle("Vectorized Image Gradient")

    v_time = image_gradient_vectorized_time()
    l_time = image_gradient_loop_time()

    plt.subplot(211).imshow(image_gradient_vectorized(colored_images[0], 5))
    plt.title("Image Gradient without using loops")
    plt.ylabel('execution time = {}'.format(round(v_time, 3)))
    plt.xticks([])
    plt.yticks([])

    plt.subplot(212).imshow(image_gradient_loop(colored_images[0], 5))
    plt.title("Image Gradient using loops")
    plt.ylabel('execution time = {}'.format(round(l_time, 3)))
    plt.xticks([])
    plt.yticks([])


# /** Source : https://www.geeksforgeeks.org/timeit-python-examples/  **/

# compute image_gradient_vectorized
def image_gradient_vectorized_time():
    SETUP_CODE = ''' 
from __main__ import image_gradient_vectorized
import matplotlib.image as mpimg
sir_pigeon = mpimg.imread("images/some-pigeon.jpg")
    '''

    TEST_CODE = '''
image_gradient_vectorized(sir_pigeon, brightness=1)
    '''
    # timeit.repeat statement
    times = timeit.repeat(setup=SETUP_CODE, stmt=TEST_CODE, number=1)

    # priniting minimum exec. time
    return min(times)


# compute image_gradient_loop
def image_gradient_loop_time():
    SETUP_CODE = ''' 
from __main__ import image_gradient_loop
import matplotlib.image as mpimg
sir_pigeon = mpimg.imread("images/some-pigeon.jpg")
    '''

    TEST_CODE = '''
image_gradient_loop(sir_pigeon, brightness=1)
    '''
    # timeit.repeat statement
    times = timeit.repeat(setup=SETUP_CODE, stmt=TEST_CODE, number=1)

    # priniting minimum exec. time
    return min(times)


def main():

    image_paths = ["images/some-pigeon.jpg", "images/colortrui.png", "images/girlWithScarf.png", "images/House.jpg",
                   "images/peppers.png", "images/Pyramids2.jpg"]
    colored_images, gray_images = load_images(image_paths)
    display_images(colored_images)

    global sir_pigeon
    sir_pigeon = colored_images[0]

    fig = plt.figure(1, figsize=(10, 6))
    fig.subplots_adjust(hspace=0.5, wspace=0.5)
    plt.suptitle("Sir Pigeon")

    plt.subplot(221).imshow(colored_images[0])
    plt.title('Colored Image')
    plt.xticks([])
    plt.yticks([])

    plt.subplot(222).imshow(gray_images[0], cmap=plt.get_cmap('gray'))
    plt.title('Gray_scale Image')
    plt.xticks([])
    plt.yticks([])

    x_range = np.arange(256)
    red_hist, green_hist, blue_hist = display_histogram(colored_images[0])
    plt.subplot(223).plot(x_range, red_hist, 'r', x_range, green_hist, 'g', x_range, blue_hist, 'b')
    plt.xlabel('Intensity')
    plt.ylabel('Frequency')
    plt.title('Image Histogram')

    plt.subplot(224).imshow(image_gradient_vectorized(colored_images[0], 5), cmap=plt.get_cmap('gray'))
    plt.title('Image Gradient')
    plt.xticks([])
    plt.yticks([])

    display_image_gradient(colored_images)
    image_stat_show(colored_images[0])

    plt.show()


if __name__ == '__main__':
    main()