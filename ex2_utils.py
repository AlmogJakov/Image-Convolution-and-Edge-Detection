import math
import numpy as np
from cv2 import cv2


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


# TODO: conv as we saw in the lecture
def conv2D(in_image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """
    Convolve a 2-D array with a given kernel
    :param in_image: 2D image
    :param kernel: A kernel
    :return: The convolved image
    """
    if kernel.ndim == 1:
        kernel = np.array([kernel]).T  # To 2D vertical vector (note the extra square brackets).
    k_row = len(kernel)
    k_col = len(kernel[0])
    half_shape = tuple(int(np.floor(i / 2)) for i in kernel.shape)
    new_img = np.pad(in_image, ((half_shape[0], half_shape[0]), (half_shape[1], half_shape[1])), mode='edge')
    # Instead of flipping the kernel twice and use the regular convolution formula
    # we can leave the kernel as is and use element-wise multiplication.
    result = [[(np.round(np.sum(new_img[i:i + k_row, j:j + k_col] * kernel))) for j in range(len(in_image[0]))]
              for i in range(len(in_image))]
    # return np.clip(np.array(result).astype('int'), 0, 255)
    # The function can return negative values (useful for laplacian)
    return np.array(result).astype('int')


def convDerivative(in_image: np.ndarray) -> (np.ndarray, np.ndarray):
    """
    Calculate gradient of an image
    :param in_image: Grayscale iamge
    :return: (directions, magnitude)
    """
    x_kernel = np.array([[-1, 0, 1]])  # Horizontal vector
    y_kernel = x_kernel.T  # Vertical vector
    x = np.clip(conv2D(in_image, x_kernel), 0, 255) / 255.0
    y = np.clip(conv2D(in_image, y_kernel), 0, 255) / 255.0
    magnitude = np.sqrt(x ** 2 + y ** 2)
    directions = np.arctan(np.divide(y, x, out=np.zeros_like(y), where=x != 0))
    return directions, magnitude


'''
2D Gaussian kernel Commonly approximated using the binomial coefficients:
E.g for vector of binomial coefficients of size three: [1, 2, 1]
     ___                 _______
    | 1 |    _______    | 1 2 1 |
    | 2 | X | 1 2 1 | = | 2 4 2 |
    | 1 |   |_______|   | 1 2 1 |
    |___|               |_______|
'''


def blurImage1(in_image: np.ndarray, k_size: int) -> np.ndarray:
    """
    Blur an image using a Gaussian kernel
    :param in_image: Input image
    :param k_size: Kernel size
    :return: The Blurred image
    """
    f = math.factorial
    bin_vector = np.array([[f(k_size - 1) // f(i) // f(k_size - 1 - i) for i in range(k_size)]])
    kernel = np.dot(bin_vector.T, bin_vector)
    normalized_kernel = kernel * (1 / np.sum(kernel)) if ((np.sum(kernel)) != 0) else np.zeros(kernel.shape)
    # return cv2.filter2D(in_image * 255.0, -1, normalized_kernel, borderType=cv2.BORDER_REPLICATE) / 255
    return conv2D(in_image * 255.0, normalized_kernel) / 255


def blurImage2(in_image: np.ndarray, k_size: int) -> np.ndarray:
    """
    Blur an image using a Gaussian kernel using OpenCV built-in functions
    :param in_image: Input image
    :param k_size: Kernel size
    :return: The Blurred image
    """
    return cv2.blur(in_image, (k_size, k_size))


def edgeDetectionZeroCrossingSimple(img: np.ndarray) -> np.ndarray:
    """
    Detecting edges using "ZeroCrossing" method
    :param img: Input image
    :return: Edge matrix
    """
    new_img = np.pad(img, ((1, 1), (1, 1)), mode='edge')
    laplacian_matrix = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
    mat = conv2D(new_img * 255.0, laplacian_matrix)
    result = [[(1 if (mat[i][j] ^ mat[i][j+2] < 0) or (mat[i][j] ^ mat[i+2][j] < 0) else 0) for j in range(len(img[0]))]
              for i in range(len(img))]
    return np.array(result).astype('float32')


def edgeDetectionZeroCrossingLOG(img: np.ndarray) -> np.ndarray:
    """
    Detecting edges using "ZeroCrossingLOG" method
    :param img: Input image
    :return: Edge matrix
    """
    new_img = blurImage1(img, 5)
    return edgeDetectionZeroCrossingSimple(new_img)


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
