# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import cv2
import numpy as np

from ex2_utils import conv1D, conv2D


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

    # img = cv2.imread('input/beach.jpg', cv2.IMREAD_GRAYSCALE)
    # kernel = np.ones((5, 5))
    # kernel = kernel / kernel.sum()
    # c_img = conv2D(img, kernel) / 255
    # cv_img = cv2.filter2D(img, -1, kernel, borderType=cv2.BORDER_REPLICATE) / 255

    # k = [[-4,-3,-2],
    #      [-1,0,1],
    #      [2, 3, 4]]

    src = np.array([[4, 3, 5, 1, 5],
                    [5, 4, 7, 1, 2],
                    [6, 4, 6, 1, 2],
                    [7, 2, 3, 5, 6]])

    src = np.array([[135 ,139 ,143 ,146 ,148 ,147 ,146 ,146 ,144 ,140],
            [135 ,138 ,142 ,145 ,147 ,146 ,145 ,145 ,143 ,139],
            [135 ,137 ,141 ,144 ,146 ,145 ,144 ,144 ,142 ,138],
            [134 ,137 ,141 ,144 ,146 ,145 ,144 ,144 ,142 ,138]])
    cv2.imwrite('Test_gray.jpg', src)
    img = cv2.imread('Test_gray.jpg', cv2.IMREAD_GRAYSCALE)
    print(img)
    # k = np.array([[1/5, 1/2, 1/7],
    #               [1/6, 1/4, 1/3]])
    k = np.ones((6, 3))
    k = k / k.sum()
    print(k)
    print(cv2.filter2D(img, -1, k, borderType=cv2.BORDER_REPLICATE))
    # half_shape = tuple(int(i / 2 + 1) for i in k.shape)[::-1]
    # print(half_shape)
    # new_img = np.pad(src, (half_shape, half_shape), mode='edge')
    half_shape = tuple(int(np.floor(i / 2)) for i in k.shape)
    new_img = np.pad(img, ((half_shape[0], half_shape[0]), (half_shape[1], half_shape[1])), mode='edge')
    print(new_img)
    print()
    print(conv2D(img, k))
    # print(new_img)
    # f_row = 1
    # f_col = 1
    # sub = new_img[0:3, 0:3]
    # print(sub)
