import math
import numpy as np
import cv2


def conv1D(in_signal: np.ndarray, k_size: np.ndarray) -> np.ndarray:
    """
    Convolve a 1-D array with a given kernel
    :param in_signal: 1-D array
    :param k_size: 1-D array as a kernel
    :return: The convolved array
    """
    dt = int
    if in_signal.dtype == float or k_size.dtype == float:
        dt = float
    reversed_k = np.flip(k_size)
    result_len = len(in_signal) + len(k_size) - 1
    result = np.zeros((result_len,), dtype=dt)
    for i in range(result_len):
        temp = np.zeros((len(in_signal) * 2,), dtype=dt)
        np.put(temp, np.arange(i + 1 - len(k_size), i + 1), reversed_k, mode='clip')
        result[i] = np.sum(np.dot(temp[0:len(in_signal)], in_signal))
    return result[0:result_len]


def conv2D(in_image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """
    Convolve a 2-D array with a given kernel
    :param in_image: 2D image
    :param kernel: A kernel
    :return: The convolved image
    """
    # k_row = len(kernel)
    # k_col = len(kernel[0])
    # image_row = len(in_image)
    # new_k = np.zeros(kernel.shape)
    # result = np.zeros(in_image.shape)
    # # kernel = np.flip(kernel, axis=0)  # flip vertically
    # # kernel = np.flip(kernel, axis=1)  # flip hor
    # for i in range(len(in_image)):
    #     for j in range(len(in_image[0])):
    #         f_row = int(i - k_row / 2 + 0.5)
    #         f_col = int(j - k_col / 2 + 0.5)
    #         indices = np.arange(f_col, f_col + k_col)
    #         t = [np.take(in_image[sorted((0, f_row + idx, image_row - 1))[1]], indices, mode='clip') for idx, item
    #              in enumerate(new_k)]
    #         result[i][j] = int(np.sum(t * kernel)+0.5)
    # return result

    # t = [np.take(in_image[sorted((0, f_row + idx, image_row - 1))[1]], indices, mode='clip') for idx, item
    #      in enumerate(new_k)]

    # t = [np.take(in_image[max(min(image_row - 1, f_row + idx), 0)], indices, mode='clip') for idx, item
    #      in enumerate(new_k)]

    k_row = len(kernel)
    k_col = len(kernel[0])
    # kernel = np.flip(kernel, axis=0)  # flip vertically
    # kernel = np.flip(kernel, axis=1)  # flip hor
    half_shape = tuple(int(np.floor(i / 2)) for i in kernel.shape)
    new_img = np.pad(in_image, ((half_shape[0], half_shape[0]), (half_shape[1], half_shape[1])), mode='edge')
    # print(np.round(round(np.sum(new_img[3:3+k_row, 2:2+k_col] * 1/18),8)))
    result = [[(np.round(np.sum(new_img[i:i + k_row, j:j + k_col] * kernel))) for j in range(len(in_image[0]))]
              for i in range(len(in_image))]
    # print([[np.sum(new_img[i:i+k_row, j:j+k_col] * kernel) for j in range(len(in_image[0]))]
    #           for i in range(len(in_image))])
    return np.clip(np.array(result).astype('int'), 0, 255)


def convDerivative(in_image: np.ndarray) -> (np.ndarray, np.ndarray):
    """
    Calculate gradient of an image
    :param in_image: Grayscale iamge
    :return: (directions, magnitude)
    """

    return


def blurImage1(in_image: np.ndarray, k_size: int) -> np.ndarray:
    """
    Blur an image using a Gaussian kernel
    :param in_image: Input image
    :param k_size: Kernel size
    :return: The Blurred image
    """

    return


def blurImage2(in_image: np.ndarray, k_size: int) -> np.ndarray:
    """
    Blur an image using a Gaussian kernel using OpenCV built-in functions
    :param in_image: Input image
    :param k_size: Kernel size
    :return: The Blurred image
    """

    return


def edgeDetectionZeroCrossingSimple(img: np.ndarray) -> np.ndarray:
    """
    Detecting edges using "ZeroCrossing" method
    :param img: Input image
    :return: opencv solution, my implementation
    """

    return


def edgeDetectionZeroCrossingLOG(img: np.ndarray) -> np.ndarray:
    """
    Detecting edges usint "ZeroCrossingLOG" method
    :param img: Input image
    :return: opencv solution, my implementation
    """

    return


def houghCircle(img: np.ndarray, min_radius: int, max_radius: int) -> list:
    """
    Find Circles in an image using a Hough Transform algorithm extension
    To find Edges you can Use OpenCV function: cv2.Canny
    :param img: Input image
    :param min_radius: Minimum circle radius
    :param max_radius: Maximum circle radius
    :return: A list containing the detected circles,
                [(x,y,radius),(x,y,radius),...]
    """

    return


def bilateral_filter_implement(in_image: np.ndarray, k_size: int, sigma_color: float, sigma_space: float) -> (
        np.ndarray, np.ndarray):
    """
    :param in_image: input image
    :param k_size: Kernel size
    :param sigma_color: represents the filter sigma in the color space.
    :param sigma_space: represents the filter sigma in the coordinate.
    :return: OpenCV implementation, my implementation
    """

    return
