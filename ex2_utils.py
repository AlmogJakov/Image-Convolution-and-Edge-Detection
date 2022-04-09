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
    if kernel.ndim == 1:
        kernel = np.array([kernel])
        kernel = kernel.reshape(kernel.shape[::-1])
    k_row = len(kernel)
    k_col = len(kernel[0])
    half_shape = tuple(int(np.floor(i / 2)) for i in kernel.shape)
    new_img = np.pad(in_image, ((half_shape[0], half_shape[0]), (half_shape[1], half_shape[1])), mode='edge')
    result = [[(np.round(np.sum(new_img[i:i + k_row, j:j + k_col] * kernel))) for j in range(len(in_image[0]))]
              for i in range(len(in_image))]
    return np.clip(np.array(result).astype('int'), 0, 255)


def convDerivative(in_image: np.ndarray) -> (np.ndarray, np.ndarray):
    """
    Calculate gradient of an image
    :param in_image: Grayscale iamge
    :return: (directions, magnitude)
    """
    y_kernel = np.array([[1, 0, -1]])
    x_kernel = y_kernel.reshape(y_kernel.shape[::-1])
    new_img = np.pad(in_image, ((1, 1), (1, 1)), mode='edge')
    # print(new_img)
    magnitude = [[math.sqrt(np.sum(new_img[i:i + 3, j + 1:j + 2] * x_kernel) ** 2 +
                            np.sum(new_img[i + 1:i + 2, j:j + 3] * y_kernel) ** 2) for j in
                  range(len(in_image[0]))]
                 for i in range(len(in_image))]
    # directions = [[np.arctan(np.sum(new_img[i + 1:i + 2, j:j + 3] * y_kernel) ** 2 /
    #                          np.sum(new_img[i:i + 3, j + 1:j + 2] * x_kernel) ** 2) for j in
    #                range(len(in_image[0]))]
    #               for i in range(len(in_image))]
    # print(magnitude)
    # print(directions)
    result = in_image
    cv2.magnitude(in_image[:1],in_image[:1], result)
    print(result)
    print()
    print(np.array(magnitude))
    return magnitude, magnitude


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
    :return: Edge matrix
    """

    return


def edgeDetectionZeroCrossingLOG(img: np.ndarray) -> np.ndarray:
    """
    Detecting edges using "ZeroCrossingLOG" method
    :param img: Input image
    :return: Edge matrix
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
