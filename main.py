# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import math

from cv2 import cv2
import numpy as np
import matplotlib.pyplot as plt

from ex2_utils import conv1D, conv2D, convDerivative, blurImage1, blurImage2, edgeDetectionZeroCrossingSimple, \
    houghCircle


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')
    print("Your OpenCV version is: " + cv2.__version__)

    # # test 1.a
    #
    # a = np.array([1, 2, 3, 4, 5])
    # b = np.array([1, 2, 3])
    # print(np.convolve(a, b, 'full'))
    # print(conv1D(a, b))
    #
    # a = np.array([1, 2, 3, 4, 5, 6])
    # b = np.array([2, 6, 10, 5, 3, 8.5, 9])
    # print(np.convolve(a, b, 'full'))
    # print(conv1D(a, b))
    # np.pad

    # src = np.array([[4, 3, 5, 1],
    #               [5, 4, 7, 1],
    #               [3, 4, 6, 1]])
    # k = np.array([[0, -1, 0],
    #               [-1, 5, -1],
    #               [0, -1, 0]])
    # src = cv2.imread("beach.jpg", 0)
    # k_row = 3
    # k_col = 3
    # new_k = np.zeros((3, 3))
    # result = np.zeros(src.shape)
    # for i in range(len(src)):
    #     for j in range(len(src[0])):
    #         f_row = int(i - 3 / 2 + 0.5)
    #         f_col = int(j - 3 / 2 + 0.5)
    #         indices = np.arange(f_col, f_col + 3)
    #         t = [np.take(src[sorted((0, f_row + idx, len(src)-1))[1]], indices, mode='clip') for idx, item in
    #              enumerate(new_k)]
    #         result[i][j] = np.sum(t * k)
    # print(result)
    # print(cv2.filter2D(src, -1, np.int8(k), borderType=cv2.BORDER_REPLICATE))

    # img = cv2.imread('test.jpg', cv2.IMREAD_GRAYSCALE)
    # convDerivative(img)
    # kernel = np.ones((5, 5))

    # kernel = kernel / kernel.sum()
    # c_img = conv2D(img, kernel) / 255
    # cv_img = cv2.filter2D(img, -1, kernel, borderType=cv2.BORDER_REPLICATE) / 255

    # k = [[-4,-3,-2],
    #      [-1,0,1],
    #      [2, 3, 4]]

    # img = cv2.imread('test.jpg', cv2.IMREAD_GRAYSCALE)
    # kernel = np.ones((5, 5))
    # kernel = kernel / kernel.sum()
    # # print(kernel)
    # # c_img = conv2D(img, kernel) / 255
    # # print(c_img)
    # # print()
    # print(img)
    # cv_img = cv2.filter2D(img, -1, kernel, borderType=cv2.BORDER_REPLICATE) / 255
    # print()
    # print(cv_img)
    # print("Max Error: {}".format(np.abs(c_img - cv_img).max() * 255))

    # img = cv2.imread('input/beach.jpg', cv2.IMREAD_GRAYSCALE)
    # # print()
    # k_size = 3
    # # b1 = blurImage1(img, k_size)
    # # b2 = blurImage2(img, k_size)
    # # print(b1)
    # # print()
    # # print(b2)
    # # print("Blurring MSE:{:.6f}".format(np.sqrt(np.power(b1 - b2, 2).mean())))
    # #
    # # f, ax = plt.subplots(1, 3)
    # # ax[0].imshow(b1)
    # # ax[1].imshow(b1 - b2)
    # # ax[2].imshow(b2)
    # # plt.show()
    #
    # blurImage1(img, k_size)

    # print((1 ^ -1))
    # img = cv2.imread('input/beach.jpg', cv2.IMREAD_GRAYSCALE)
    # edgeDetectionZeroCrossingSimple(img)

    # img = cv2.imread('input/cln.jpg', cv2.IMREAD_GRAYSCALE)
    # houghCircle(img, 0 ,0)

    # mat = np.zeros((20, 20))
    # x = 4
    # y = 4
    # min_radius = 3
    # max_radius = 4
    # for radius in range(min_radius, max_radius + 1):
    #     diameter = 2 * radius + 1
    #     start_x = x - radius
    #     start_y = y - radius
    #     for i in range(max(0, start_x), min(len(mat) - 1, start_x + diameter)):
    #         for j in range(max(0, start_y), min(len(mat[0]) - 1, start_y + diameter)):
    #             if np.floor(np.sqrt((i - x) ** 2 + (j - y) ** 2) + 0.5) == radius:
    #                 mat[i][j] = radius
    # print(mat)

    img = np.zeros((20, 20))
    img[1][2] = img[1][3] = img[1][4] = 1
    img[5][2] = img[5][3] = img[5][4] = 1
    img[2][1] = img[3][1] = img[4][1] = 1
    img[2][5] = img[3][5] = img[4][5] = 1
    img = cv2.imread('input/coins.jpg', cv2.IMREAD_GRAYSCALE)
    img = cv2.Canny(img, 175, 175) / 255
    min_radius = 2
    max_radius = 3
    directions, magnitude = convDerivative(img)
    #print(directions)
    np.set_printoptions(threshold=np.inf)
    print(directions * 255)
    f, ax = plt.subplots(1, 2)
    ax[0].set_title('Ori')
    ax[1].set_title('Mag')
    ax[0].imshow(directions)
    ax[1].imshow(magnitude)
    plt.show()

    # #img = cv2.imread('testcircle.jpg', cv2.IMREAD_GRAYSCALE)
    # # compute gradients along the x and y axis, respectively
    # gX = cv2.Sobel(img, cv2.CV_64F, 1, 0)
    # gY = cv2.Sobel(img, cv2.CV_64F, 0, 1)
    # # compute the gradient magnitude and orientation
    # magnitude = np.sqrt((gX ** 2) + (gY ** 2))
    # orientation = np.arctan2(gY, gX) * (180 / np.pi) % 180
    # print(orientation)
    # plt.imshow(orientation)
    # plt.show()


    # result = np.zeros((len(img), len(img[0]), max_radius + 1))

    # mat = np.zeros((20, 20))
    # x = 3
    # y = 3
    # for x in range(len(img)):
    #     for y in range(len(img[0])):
    #         for radius in range(min_radius, max_radius + 1):
    #             diameter = 2 * radius + 1
    #             start_x = x - radius
    #             start_y = y - radius
    #             for i in range(max(0, start_x), min(len(img) - 1, start_x + diameter)):
    #                 for j in range(max(0, start_y), min(len(img[0]) - 1, start_y + diameter)):
    #                     if np.floor(np.sqrt((i - x) ** 2 + (j - y) ** 2) + 0.5) == radius:
    #                         # mat[i][j] = radius
    #                         result[i][j][radius] = result[i][j][radius] + 1
    # # print(mat)
    # print()
    # for x in range(len(img)):
    #     for y in range(len(img[0])):
    #         for z in range(radius + 1):
    #             if result[x][y][z] >= np.floor(2 * np.pi * radius):
    #                 print(result[x][y][z])