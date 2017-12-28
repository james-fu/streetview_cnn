import numpy as np
import cv2



def create_pyramid(img, layers=3): #Create image pyramid for sliding window
    pyramid = []
    for i in range(layers):
        resize = 1 / (2 ** i)
        temp_image = np.copy(cv2.resize(img, (0, 0), fx=resize, fy=resize))
        pyramid.append(temp_image)

    return np.array(pyramid)


def window_cutouts(pyramid_imgs, size = 48, stride = 12):

    # R is the inverse pyramid ratio
    # This function creates a multilayer pyramid and returns bounding boxes with
    # coordinates for the original size image

    nn_size = 48  #Size that neural network will use as input
    wndw_imgs = []
    wndw_loc = []


    # Find cutouts of each possible image to test
    for x, bgr_img in enumerate(pyramid_imgs):
        stride -= 2 # Reduce Stride in smaller images
        R = (2 ** x)
        h, w, channels = bgr_img.shape

        for i in range(0, h - size, stride):
            for j in range(0, w - size, stride):
                temp_img = cv2.resize((bgr_img[i:i + size, j:j + size]), (nn_size, nn_size))
                wndw_imgs.append(temp_img)
                wndw_loc.append([R*i, R*j, R*size])

    return np.array(wndw_imgs), np.array(wndw_loc)



if __name__ == "__main__":

    numpy_save = False
    img_file = 'samples/report.png'

    #Read Image, Turn to grayscale and subtract mean
    image = np.array(cv2.imread(img_file))
    temp_img = np.copy(image) - np.mean(image)
    pyramid = create_pyramid(temp_img)
    wndw_imgs, wndw_loc = window_cutouts(pyramid)


    if numpy_save:
        np.save("wndw_imgs.npy", wndw_imgs)
        np.save("wndw_loc.npy", wndw_loc)
