from scipy.signal import convolve2d
import numpy as np
from imageio import imread
from skimage.color import rgb2gray
from scipy.ndimage.filters import convolve
from scipy import signal

FILE_PROBLEM = "File Problem"

GREYSCALE = 1

MAX_INTENSITY = 255

EVEN_INDEX = 2

MIN_RESOLUTION = 16


def gaussian_kernel(kernel_size):
    conv_kernel = np.array([1, 1], dtype=np.float64)[:, None]
    conv_kernel = convolve2d(conv_kernel, conv_kernel.T)
    kernel = np.array([1], dtype=np.float64)[:, None]
    for i in range(kernel_size - 1):
        kernel = convolve2d(kernel, conv_kernel, 'full')
    return kernel / kernel.sum()


def blur_spatial(img, kernel_size):
    kernel = gaussian_kernel(kernel_size)
    blur_img = np.zeros_like(img)
    if len(img.shape) == 2:
        blur_img = convolve2d(img, kernel, 'same', 'symm')
    else:
        for i in range(3):
            blur_img[..., i] = convolve2d(img[..., i], kernel, 'same', 'symm')
    return blur_img


def normalize_0_to_1(im):
    """
    normalize picture
    :param im: image in range 0-255
    :return: image in range [0,1]
    """
    if im.dtype != np.float64:
        im = im.astype(np.float64)
        im /= MAX_INTENSITY
    return im


def read_image(filename, representation):
    """
    This function returns an image, make sure the output image is represented by a matrix of type
    np.float64 with intensities (either grayscale or RGB channel intensities) normalized to the range [0, 1].
    :param filename: the filename of an image on disk (could be grayscale or RGB).
    :param representation: representation code, either 1 or 2 defining whether the output should be a grayscale
    image (1) or an RGB image (2).
    :return: an image
    """
    im = None
    try:
        im = imread(filename)
    except Exception:  # internet didnt have specific documentation regarding the exceptions this func throws
        print(FILE_PROBLEM)
        exit()
    im = normalize_0_to_1(im)
    if representation == GREYSCALE:
        return rgb2gray(im)
    return im.astype(np.float64)


def reduce(im, filter):
    """
    Blur & sub-sample
    :param im: im to reduce
    :param filter: filter to blur with using convolution
    :return: reduced im
    """
    im = convolve(im, filter)
    im = convolve(im, filter.T)
    return im[:im.shape[0]:EVEN_INDEX, :im.shape[1]:EVEN_INDEX]


def create_filter(size):
    """
    give gaussian filter of given size
    :param size: filter size
    :return: row vector
    """
    if size == 1:
        return np.array([1]).reshape((1, 1))
    base_filter = np.array([1, 1])
    filter = np.array([1, 1])
    for i in range(size - 2):
        filter = signal.convolve(filter, base_filter)
    return (filter / np.sum(filter)).reshape((1, size))


def build_gaussian_pyramid(im, max_levels, filter_size):
    """
    construct a Gaussian pyramid of a given image
    :param im: a grayscale image with double values in [0, 1]
    :param max_levels: the maximal number of levels in the resulting pyramid
    :param filter_size:  the size of the Gaussian filter (an odd scalar that represents a squared filter) to be used
            in constructing the pyramid filter
    :return: pyr: a standard python array with maximum length of max_levels, where each element
            of the array is a grayscale image,
            filter_vec: row vector of shape (1, filter_size) used for the pyramid construction
    """
    filter = create_filter(filter_size).reshape(1, filter_size)
    gaussian_pyr = [im]
    for i in range(max_levels - 1):
        curr_level = reduce(gaussian_pyr[i], filter)
        if curr_level.shape[0] < MIN_RESOLUTION or curr_level.shape[1] < MIN_RESOLUTION:
            break
        gaussian_pyr.append(curr_level)
    return gaussian_pyr, filter

def expand(im, filter):
    """
    pad with zeroes & blur
    :param im: im to expand
    :param filter: filter to blur with using convolution
    :return: expanded im
    """
    new_im = np.zeros((im.shape[0] * 2, im.shape[1] * 2))
    new_im[:new_im.shape[0]:EVEN_INDEX, :new_im.shape[1]:EVEN_INDEX] = im
    new_im = convolve(new_im, 2 * filter)
    new_im = convolve(new_im, 2 * filter.T)
    return new_im

def build_laplacian_pyramid(im, max_levels, filter_size):
    """
    construct a Laplacian pyramid of a given image
    :param im: a grayscale image with double values in [0, 1]
    :param max_levels: the maximal number of levels in the resulting pyramid
    :param filter_size:  the size of the Gaussian filter (an odd scalar that represents a squared filter) to be used
            in constructing the pyramid filter
    :return: pyr: a standard python array with maximum length of max_levels, where each element
            of the array is a grayscale image,
            filter_vec: row vector of shape (1, filter_size) used for the pyramid construction
    """
    gaussian_pyr, filter = build_gaussian_pyramid(im, max_levels, filter_size)
    laplacian_pyr = []
    for i in range(len(gaussian_pyr) - 1):
        laplacian_pyr.append(gaussian_pyr[i] - expand(gaussian_pyr[i + 1], filter))
    laplacian_pyr.append(gaussian_pyr[-1])
    return laplacian_pyr, filter

def laplacian_to_image(lpyr, filter_vec, coeff):
    """
    e reconstruction of an image from its Laplacian Pyramid
    :param lpyr: a standard python array, where each element of the array is a grayscale image
    :param filter_vec: row vector of shape (1, filter_size)
    :param coeff: python list
    :return: reconstructed image
    """
    for i in range(len(lpyr)):
        lpyr[i] = lpyr[i] * coeff[i]
    for i in range(1, len(lpyr)):
        lpyr[-i - 1] += expand(lpyr[-i], filter_vec)
    return lpyr[0]

def pyramid_blending(im1, im2, mask, max_levels, filter_size_im, filter_size_mask):
    """
     pyramid blending
    :param im1: input grayscale image to be blended
    :param im2: input grayscale image to be blended
    :param mask: a boolean mask containing True and False representing which parts
            of im1 and im2 should appear in the resulting im_blend. Note that a value of True corresponds to 1,
            and False corresponds to 0.
    :param max_levels: is the max_levels parameter when generating the Gaussian and Laplacian pyramids
    :param filter_size_im: is the size of the Gaussian filter (an odd scalar that represents a squared filter) which
            defining the filter used in the construction of the Laplacian pyramids of im1 and im2
    :param filter_size_mask: is the size of the Gaussian filter(an odd scalar that represents a squared filter) which
            defining the filter used in the construction of the Gaussian pyramid of mask.
    :return: grayscale image
    """
    L_1, filter_vec_im = build_laplacian_pyramid(im1, max_levels, filter_size_im)
    L_2, _ = build_laplacian_pyramid(im2, max_levels, filter_size_im)
    G_m, filter_vec_mask = build_gaussian_pyramid(mask.astype(np.float64), max_levels, filter_size_mask)
    L_out = []
    for i in range(len(L_1)):
        L_out.append(G_m[i] * L_1[i] + (1 - G_m[i]) * L_2[i])
    coeff = [1] * len(L_out)
    return laplacian_to_image(L_out, filter_vec_im, coeff).clip(min=0, max=1)
