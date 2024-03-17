import imageio.v3 as iio
import numpy as np
from typing import Dict

def get_input(low_input: str, high_input: str):
    """
    Function to read input images.

    Args:
    - low_input: str, the path prefix for the low-resolution input images.
    - high_input: str, the path for the high-resolution input image.

    Returns:
    - img_low: dict, a dictionary containing the low-resolution input images.
    - img_high: np.array, the high-resolution input image.
    """
    img_low = {}
    for i in range(0,4):
        img_low[i] = iio.imread(f"test_cases/{low_input}{i}.png")

    img_high = iio.imread(f"test_cases/{high_input}")
    
    return img_low, img_high


def rmse(img_high: np.array, img_high_calculated: np.array, print_error: bool = True):
    """
    Function to calculate the Root Mean Square Error (RMSE) between two images.

    Args:
    - img_high: np.array, the ground truth high-resolution image.
    - img_high_calculated: np.array, the calculated high-resolution image.
    - print_error: bool, whether to print the error value or not (default: True).

    Returns:
    - error: float, the RMSE value.
    """
    error = np.sqrt(((img_high - img_high_calculated)**2).sum()/img_high.size)
    
    if print_error is True:
        print(f"{error:.4f}") 
    return error


def intercalate(img1: np.array,img2: np.array):
    """
    Function to intercalate two images horizontally.
    
    Args:
    - img1: np.array, the first image.
    - img2: np.array, the second image.

    Returns:
    - img3: np.array, the intercalated image.
    """
    N, M = img1.shape
    img3 = np.empty( [N, M+M], dtype=img1.dtype)

    for row in range(0, img3.shape[0]):
        img3[row][0::2] = img1[row]
        img3[row][1::2] = img2[row]

    return img3
    
    
def superresolution(img_dict: Dict[int, np.array]):
    """
    Function to perform superresolution by intercalating and stacking two images.

    Args:
    - img_dict: dict, a dictionary containing the input images.

    Returns:
    - img3: np.array, the superresolved image.
    """
    img1 = intercalate(img_dict[0],img_dict[2])
    img2 = intercalate(img_dict[1],img_dict[3])

    N, M = img1.shape
    img3 = np.empty([N+N, M], dtype= np.dtype('uint32'))
    img3[0::2] = img1
    img3[1::2] = img2
    
    return img3


def histogram(img: np.array, n_levels: int):
    """
    Function to calculate the histogram of an image.

    Args:
    - img: np.array, the input image.
    - n_levels: int, the number of intensity levels.

    Returns:
    - hist: np.array, the histogram of the image.
    """
    hist = np.empty(n_levels, dtype=int)
    for level in range(n_levels):
        hist[level] = np.sum(img == level)
        
    return hist


def cumulative_histogram(
    n_levels: int = 256, 
    joint: bool = False, 
    img: np.array = None,
    img_dict: Dict[int, np.array] = None):
    """
    Function to calculate the cumulative histogram of an image or a set of images.

    Args:
    - n_levels: int, the number of intensity levels (default: 256).
    - joint: bool, whether to calculate the joint cumulative histogram or not (default: False).
    - img: np.array, the input image (required if joint is False).
    - img_dict: dict, a dictionary containing the input images (required if joint is True).

    Returns:
    - histC: np.array, the cumulative histogram.
    - resol: float, the resolution of the image(s).
    """
    N = M = n_levels
    if joint is True:
        hist = np.zeros(n_levels, dtype=int)
        for key in img_dict:
            hist = hist + histogram(img_dict[key], n_levels)
        resol = float(N*M*4)
    else:
        hist = histogram(img, n_levels)
        resol = float(N*M)
        
    histC = np.empty(n_levels, dtype=int)
    
    histC[0] = hist[0]
    for i in range(1, n_levels):
        histC[i] = hist[i] + histC[i-1]  
        
    return histC, resol


def histogram_equalization(
        img: np.array, histC: np.array, resol: int,
        n_levels: int = 256):
    """
    Function to perform histogram equalization on an image.

    Args:
    - img: np.array, the input image.
    - histC: np.array, the cumulative histogram.
    - resol: int, the resolution of the image.
    - n_levels: int, the number of intensity levels (default: 256).

    Returns:
    - new_img: np.array, the histogram equalized image.
    """
    new_img = np.empty([n_levels,n_levels], dtype = np.dtype('uint32'))
    for level in range(n_levels):     
        new_img[np.where(img == level)] = (n_levels-1)*histC[level]/resol
    return new_img


def single_image_cumulative_histogram(img_dict: Dict[int, np.array]):
    """
    Function to perform enhancement using single image cumulative histogram equalization.

    Args:
    - img_dict: dict, a dictionary containing the low-resolution input images.

    Returns:
    - new_img: np.array, the enhanced image.
    """
    new_img_dict = {}
    for key in img_dict:
        histC, resol = cumulative_histogram(img=img_dict[key])
        new_img_dict[key] = histogram_equalization(img_dict[key], histC, resol)
        
    new_img = superresolution(new_img_dict)
    return new_img


def joint_cumulative_histogram(img_dict: Dict[int, np.array]):
    """
    Function to perform enhancement using joint cumulative histogram equalization.

    Args:
    - img_dict: dict, a dictionary containing the low-resolution input images.

    Returns:
    - new_img: np.array, the enhanced image.
    """
    histC, resol = cumulative_histogram(img_dict=img_dict, joint=True)
    
    new_img_dict = {}
    for key in img_dict:
        new_img_dict[key] = histogram_equalization(img_dict[key], histC, resol)
        
    new_img = superresolution(new_img_dict)
    return new_img


def gamma_function(img: np.array, gamma: float):
    """
    Function to calculate the gamma correction on an image.

    Args:
    - img: np.array, the input image.
    - gamma: float, the gamma value.

    Returns:
    - new_img: np.array, the gamma corrected image.
    """
    return 255 * ((img/255.0)**(1.0/gamma))


def gamma_correction(img_dict: Dict[int, np.array], gamma: float):
    """
    Function to perform enhancement using gamma correction.

    Args:
    - img_dict: dict, a dictionary containing the low-resolution input images.
    - gamma: float, the gamma value.

    Returns:
    - new_img: np.array, the enhanced image.
    """
    new_img_dict = {}
    for key in img_dict:
        new_img_dict[key] = gamma_function(img_dict[key], gamma)
    
    new_img = superresolution(new_img_dict)
    return new_img