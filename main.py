# This is a sample Python script.
from cv2 import cv2
# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import math
import numpy as np
from cv2 import cv2
import matplotlib.pyplot as plt

from ex2_utils import conv2D


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')
    print("Your OpenCV version is: " + cv2.__version__)
    # img = np.array([[0, -1, 0], [0, 0, 0], [0, 1, 0]])
    # kernel = np.array([[0, 0, 0], [-1, 0, 1], [0, 0, 0]])
    img = np.array([[0, 1, 0],
                    [0, 2, 0],
                    [0, 1, 0]])
    kernel = np.array([[0, 0, 0],
                       [-1, 0, 1],
                       [0, 0, 0]])
    #cv2.filter2D(img, -1, kernel, borderType=cv2.BORDER_REPLICATE)
    print(conv2D(img, kernel))