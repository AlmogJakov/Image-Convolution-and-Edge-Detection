# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import cv2
import numpy as np

from ex2_utils import conv1D, conv2D, convDerivative


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

    img = cv2.imread('test.jpg', cv2.IMREAD_GRAYSCALE)
    convDerivative(img)
    kernel = np.ones((5, 5))
    # kernel = kernel / kernel.sum()
    # c_img = conv2D(img, kernel) / 255
    # cv_img = cv2.filter2D(img, -1, kernel, borderType=cv2.BORDER_REPLICATE) / 255

    # k = [[-4,-3,-2],
    #      [-1,0,1],
    #      [2, 3, 4]]


