import math
import numpy as np
from cv2 import cv2
import matplotlib.pyplot as plt


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


'''
'conv2D' Method:
    Instead of flipping the kernel twice and use the regular convolution formula
    we can leave the kernel as is and use element-wise multiplication.
    
    OpenCV Implementation: https://github.com/opencv/opencv/blob/master/modules/imgproc/src/opencl/filter2D.cl
'''


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
    # The function can return negative values (useful for laplacian)
    return np.array(result)


'''
'convDerivative' Method:
    About 'arctan2':
        in case of x=0: arctan2(y, x) returns theta = -pi or pi
        Note! 'np.arctan2' returns Radian values!
        And hence we want to returns degree values -> we use 'rad2deg' function. (multiply by 180 / pi).
        check more here: http://library.isr.ist.utl.pt/docs/numpy/reference/generated/numpy.arctan2.html
        
    more sources: 
        https://lilianweng.github.io/posts/2017-10-29-object-recognition-part-1/
        https://docs.opencv.org/4.x/d5/d0f/tutorial_py_gradients.html
        https://stackoverflow.com/questions/19815732/what-is-the-gradient-orientation-and-gradient-magnitude
'''


def convDerivative(in_image: np.ndarray) -> (np.ndarray, np.ndarray):
    """
    Calculate gradient of an image
    :param in_image: Grayscale iamge
    :return: (directions, magnitude)
    """
    x_kernel = np.array([[1, 0, -1]])  # Horizontal vector
    y_kernel = x_kernel.T  # Vertical vector
    x = conv2D(in_image * 255.0, x_kernel) / 255.0
    y = conv2D(in_image * 255.0, y_kernel) / 255.0
    directions = np.rad2deg(np.arctan2(y, x).astype(np.float64))
    magnitude = np.sqrt(x ** 2 + y ** 2).astype(np.float64)
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
    # https://docs.opencv.org/3.4/da/d5c/tutorial_canny_detector.html TODO: fix Gaussian kernel
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
    # result = [[(1 if (mat[i][j] > 0 and mat[i][j + 2] < 0) or (mat[i][j] > 0 and mat[i + 2][j] < 0) else 0) for j in
    #            range(len(img[0]))] for i in range(len(img))]
    # result = [
    #     [(1 if (((mat[i + 1][j] > 0) and (mat[i + 1][j + 2] < 0) and (np.abs(mat[i + 1][j] - mat[i + 1][j + 2] > 10)))
    #             or ((mat[i][j + 1] > 0) and (mat[i + 2][j + 1] < 0) and (
    #                 np.abs(mat[i][j + 1] - mat[i + 2][j + 1] > 10)))) else 0) for j in
    #      range(len(img[0]))] for i in range(len(img))]

    result = [
        [(1 if ((mat[i + 1][j + 1] > 0) and (mat[i + 1][j + 2] < 0))
               or ((mat[i + 1][j + 1] > 0) and (mat[i + 1][j] < 0))
               or ((mat[i + 1][j + 1] > 0) and (mat[i + 2][j + 1] < 0))
               or ((mat[i + 1][j + 1] > 0) and (mat[i][j + 1] < 0))
          else 0) for j in
         range(len(img[0]))] for i in range(len(img))]

    return np.array(result).astype('float32')


def edgeDetectionZeroCrossingLOG(img: np.ndarray) -> np.ndarray:
    """
    Detecting edges using "ZeroCrossingLOG" method
    :param img: Input image
    :return: Edge matrix
    """
    new_img = blurImage2(img, 2)
    return edgeDetectionZeroCrossingSimple(new_img)


'''
'houghCircle' Method:
    Algorithm: https://theailearner.com/tag/hough-gradient-method/
    sobel: https://pyimagesearch.com/2021/05/12/image-gradients-with-opencv-sobel-and-scharr/
    sobel: http://www.adeveloperdiary.com/data-science/computer-vision/how-to-implement-sobel-edge-detection-using-python-from-scratch/
    sobel: https://docs.opencv.org/3.4/d2/d2c/tutorial_sobel_derivatives.html
    https://pyimagesearch.com/2021/05/12/image-gradients-with-opencv-sobel-and-scharr/
    https://github.com/ido1Shapira/Image_Processing_ex2/blob/master/ex2_utils.py
    OpenCV Implementation: https://github.com/opencv/opencv/blob/master/modules/imgproc/src/hough.cpp
'''


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
    img = cv2.Canny((img * 255).astype(np.uint8), 175, 175) / 255
    plt.imshow(img, cmap='gray')
    plt.show()
    circles = np.zeros((len(img), len(img[0]), max_radius + 1))

    # for x in range(len(img)):
    #     for y in range(len(img[0])):
    #         for radius in range(min_radius, max_radius + 1):
    #             if img[x][y] == 1:
    #                 diameter = 2 * radius + 1
    #                 start_x = x - radius
    #                 start_y = y - radius
    #                 for i in range(max(0, start_x), min(len(img) - 1, start_x + diameter)):
    #                     for j in range(max(0, start_y), min(len(img[0]) - 1, start_y + diameter)):
    #                         if np.floor(np.sqrt((i - x) ** 2 + (j - y) ** 2) + 0.5) == radius:
    #                             # mat[i][j] = radius
    #                             circles[i][j][radius] = circles[i][j][radius] + 1
    # result = []
    # print("f")
    # for x in range(len(img)):
    #     for y in range(len(img[0])):
    #         for z in range(min_radius, max_radius + 1):
    #             if circles[x][y][z] >= np.floor(2 * np.pi * z / 2):
    #                 print(circles[x][y][z])
    #                 result.append(x, y, z)
    directions, magnitude = convDerivative(img)
    magnitude = magnitude * 255
    circles = np.zeros((len(img), len(img[0]), max_radius + 1))
    for r in range(min_radius, max_radius + 1):
        # print("g")
        for x in range(len(img)):
            # print("gg")
            for y in range(len(img[0])):
                # for t in range(0, 360):
                t = magnitude[x][y]
                b = y - r * np.sin(t * np.pi / 180)  # polar coordinate for center(convert to radians)
                a = x - r * np.cos(t * np.pi / 180)  # polar coordinate for center(convert to radians)
                if a < 0 or a > len(img) - 1 or b < 0 or b > len(img[0]) - 1:
                    continue
                circles[int(a)][int(b)][int(r)] = circles[int(a)][int(b)][int(r)] + 1

    result = []
    print("f")
    for x in range(len(img)):
        for y in range(len(img[0])):
            for z in range(min_radius, max_radius + 1):
                if circles[x][y][z] > 2:
                    print(circles[x][y][z])
                    result.append([x, y, z])
    return result


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
