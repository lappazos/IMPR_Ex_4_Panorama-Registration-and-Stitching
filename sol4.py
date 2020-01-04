# Initial code for ex4.
# You may change this code, but keep the functions' signatures
# You can also split the code to multiple files as long as this file's API is unchanged 

import numpy as np
import os
import matplotlib.pyplot as plt
from shutil import rmtree

from scipy.ndimage.morphology import generate_binary_structure
from scipy.ndimage.filters import maximum_filter
from scipy.ndimage import label, center_of_mass, map_coordinates
import sol4_utils
import scipy.signal

HOMOGRAPHY_MAT_DIM = 3

IM_WIDTH_INDEX = 1

POINTS_FOR_RIGID = 2

POINTS_FOR_TRANSLATION = 1

Z = 2

SECOND_MAX = -2

Y = 1

X = 0

KERNEL_SIZE = 3

N = 7

M = 7

DESC_RAD = 3

PYR_LVL_TO_USE = 2

ORIGINAL_IMAGE = 0

der_vec = np.array([1, 0, -1]).reshape(1, 3)

K = 0.04

MIN_SCORE = 0.5


def harris_corner_detector(im):
    """
    Detects harris corners.
    Make sure the returned coordinates are x major!!!
    :param im: A 2D array representing an image.
    :return: An array with shape (N,2), where ret[i,:] are the [x,y] coordinates of the ith corner points.
    """
    dx = scipy.signal.convolve2d(im, der_vec, mode='same', boundary='symm')
    dy = scipy.signal.convolve2d(im, der_vec.T, mode='same', boundary='symm')
    blur_ix_squared = sol4_utils.blur_spatial(dx * dx, KERNEL_SIZE)
    blur_iy_squared = sol4_utils.blur_spatial(dy * dy, KERNEL_SIZE)
    blur_ix_iy = sol4_utils.blur_spatial(dx * dy, KERNEL_SIZE)
    det_m = blur_ix_squared * blur_iy_squared - blur_ix_iy * blur_ix_iy
    trace_m = blur_iy_squared + blur_ix_squared
    R = det_m - K * np.square(trace_m)
    R = non_maximum_suppression(R)
    return np.flip(np.argwhere(R).reshape(-1, 2), axis=1)


def sample_descriptor(im, pos, desc_rad):
    """
    Samples descriptors at the given corners.
    :param im: A 2D array representing an image.
    :param pos: An array with shape (N,2), where pos[i,:] are the [x,y] coordinates of the ith corner point.
    :param desc_rad: "Radius" of descriptors to compute.
    :return: A 3D array with shape (N,K,K) containing the ith descriptor at desc[i,:,:].
    """
    square_size = desc_rad * 2 + 1
    descriptors = np.empty((len(pos), square_size, square_size))
    for coordinate in range(len(pos)):
        x = np.linspace(pos[coordinate][X] - desc_rad, pos[coordinate][X] + desc_rad, square_size)
        y = np.linspace(pos[coordinate][Y] - desc_rad, pos[coordinate][Y] + desc_rad, square_size)
        xv, yv = np.meshgrid(x, y)
        # stack in flip
        coordinates_float = np.stack((yv, xv), axis=0)
        descriptor = map_coordinates(im, coordinates_float, order=1, prefilter=False)
        mean = np.mean(descriptor)
        temp = descriptor - mean
        norm = np.linalg.norm(temp)
        descriptors[coordinate, :, :] = temp / np.where(norm == 0, 1, norm)
    return descriptors


def get_pyramid_coordinates(pos, input_level, output_level):
    """
    adjust coordinates between pyramids
    :param pos: coordinated in input level
    :param input_level: input level index
    :param output_level: output level index
    :return: adjusted pyramid coordinates
    """
    return pow(2, input_level - output_level) * pos


def find_features(pyr):
    """
    Detects and extracts feature points from a pyramid.
    :param pyr: Gaussian pyramid of a grayscale image having 3 levels.
    :return: A list containing:
                1) An array with shape (N,2) of [x,y] feature location per row found in the image.
                   These coordinates are provided at the pyramid level pyr[0].
                2) A feature descriptor array with shape (N,K,K)
    """
    # todo: check which radius to send here
    corners = spread_out_corners(pyr[ORIGINAL_IMAGE], M, N, DESC_RAD * pow(2, PYR_LVL_TO_USE))
    descriptors = sample_descriptor(pyr[PYR_LVL_TO_USE],
                                    get_pyramid_coordinates(corners, ORIGINAL_IMAGE, PYR_LVL_TO_USE),
                                    DESC_RAD)
    return [corners, descriptors]


def match_features(desc1, desc2, min_score):
    """
    Return indices of matching descriptors.
    :param desc1: A feature descriptor array with shape (N1,K,K).
    :param desc2: A feature descriptor array with shape (N2,K,K).
    :param min_score: Minimal match score.
    :return: A list containing:
                1) An array with shape (M,) and dtype int of matching indices in desc1.
                2) An array with shape (M,) and dtype int of matching indices in desc2.
    """
    score_matrix = desc1.reshape(len(desc1), -1) @ desc2.reshape(len(desc2), -1).T
    first_axis_max = np.partition(score_matrix, SECOND_MAX, axis=0)[SECOND_MAX, :]
    second_axis_max = np.partition(score_matrix, SECOND_MAX, axis=1)[:, SECOND_MAX]
    second_biggest_second_axis = score_matrix >= second_axis_max.reshape(-1, 1)
    second_biggest_first_axis = (score_matrix.T >= first_axis_max.reshape(-1, 1)).T
    final_matrix = np.logical_and(np.logical_and(second_biggest_first_axis, second_biggest_second_axis),
                                  score_matrix > min_score)
    match_points = np.argwhere(final_matrix)
    return [match_points[:, 0].astype(int), match_points[:, 1].astype(int)]


def apply_homography(pos1, H12):
    """
    Apply homography to inhomogenous points.
    :param pos1: An array with shape (N,2) of [x,y] point coordinates.
    :param H12: A 3x3 homography matrix.
    :return: An array with the same shape as pos1 with [x,y] point coordinates obtained from transforming pos1 using H12.
    """
    pos1 = np.concatenate((pos1, np.ones(pos1.shape[0]).reshape(-1, 1)), axis=1)
    x2_y2_z2 = H12 @ pos1.T
    return (x2_y2_z2[:Z, :] / x2_y2_z2[Z, :]).T


def ransac_homography(points1, points2, num_iter, inlier_tol, translation_only=False):
    """
    Computes homography between two sets of points using RANSAC.
    :param points1: An array with shape (N,2) containing N rows of [x,y] coordinates of matched points in image 1.
    :param points2: An array with shape (N,2) containing N rows of [x,y] coordinates of matched points in image 2.
    :param num_iter: Number of RANSAC iterations to perform.
    :param inlier_tol: inlier tolerance threshold.
    :param translation_only: see estimate rigid transform
    :return: A list containing:
                1) A 3x3 normalized homography matrix.
                2) An Array with shape (S,) where S is the number of inliers,
                    containing the indices in pos1/pos2 of the maximal set of inlier matches found.
    """
    max_inliers = [None, 0]
    for iteration in range(num_iter):

        if translation_only:
            rand_indices = np.random.choice(len(points1), POINTS_FOR_TRANSLATION)
        else:
            rand_indices = np.random.choice(len(points1), POINTS_FOR_RIGID)
        H12 = estimate_rigid_transform(np.take(points1, rand_indices, axis=0), np.take(points2, rand_indices, axis=0),
                                       translation_only)
        p2_tag = apply_homography(points1, H12)
        distance = np.square(np.linalg.norm(p2_tag - points2, axis=1))
        inliers_indices_bool = distance < inlier_tol
        inliers_num = np.sum(inliers_indices_bool)
        if inliers_num > max_inliers[1]:
            max_inliers = [inliers_indices_bool, inliers_num]
    inliers_indices = np.argwhere(max_inliers[0]).reshape((-1,))
    H12_final = estimate_rigid_transform(np.take(points1, inliers_indices, axis=0),
                                         np.take(points2, inliers_indices, axis=0), translation_only)
    return [H12_final, inliers_indices.reshape(-1, )]


def display_matches(im1, im2, points1, points2, inliers):
    """
    Dispalay matching points.
    :param im1: A grayscale image.
    :param im2: A grayscale image.
    :parma pos1: An aray shape (N,2), containing N rows of [x,y] coordinates of matched points in im1.
    :param pos2: An aray shape (N,2), containing N rows of [x,y] coordinates of matched points in im2.
    :param inliers: An array with shape (S,) of inlier matches.
    """
    im = np.hstack((im1, im2))
    points2[:, 0] += im1.shape[IM_WIDTH_INDEX]
    plt.imshow(im, cmap='gray')
    points1_inliers = np.take(points1, inliers, axis=0)
    points2_inliers = np.take(points2, inliers, axis=0)
    points1_outliers = np.delete(points1, inliers, axis=0)
    points2_outliers = np.delete(points2, inliers, axis=0)
    for i in range(len(points1_outliers)):
        plt.plot([points1_outliers[i][0], points2_outliers[i][0]], [points1_outliers[i][1], points2_outliers[i][1]],
                 mfc='r', c='b', lw=.4, ms=1, marker='o', markeredgewidth=0.0)
    for i in range(len(inliers)):
        plt.plot([points1_inliers[i][0], points2_inliers[i][0]], [points1_inliers[i][1], points2_inliers[i][1]],
                 mfc='r', c='y', lw=.4, ms=1, marker='o', markeredgewidth=0.0)
    plt.show()


def accumulate_homographies(H_succesive, m):
    """
    Convert a list of succesive homographies to a
    list of homographies to a common reference frame.
    :param H_successive: A list of M-1 3x3 homography
      matrices where H_successive[i] is a homography which transforms points
      from coordinate system i to coordinate system i+1.
    :param m: Index of the coordinate system towards which we would like to
      accumulate the given homographies.
    :return: A list of M 3x3 homography matrices,
      where H2m[i] transforms points from coordinate system i to coordinate system m
    """
    result = [np.eye(HOMOGRAPHY_MAT_DIM)]
    for i in range(m - 1, -1, -1):
        H = result[0] @ H_succesive[i]
        result = [H / H[2, 2]] + result
    for i in range(m, len(H_succesive)):
        H = result[-1] @ np.linalg.inv(H_succesive[i])
        result.append(H / H[2, 2])
    # todo: to remove
    if len(result) != len(H_succesive) + 1:
        raise Exception()
    return result


def compute_bounding_box(homography, w, h):
    """
    computes bounding box of warped image under homography, without actually warping the image
    :param homography: homography
    :param w: width of the image
    :param h: height of the image
    :return: 2x2 array, where the first row is [x,y] of the top left corner,
     and the second row is the [x,y] of the bottom right corner
    """
    corners = np.array([[0, 0], [0, h - 1], [w - 1, 0], [w - 1, h - 1]])
    new_corners = apply_homography(corners, homography)
    return np.array([[np.min(new_corners[:, 0]), np.min(new_corners[:, 1])],
                     [np.max(new_corners[:, 0]), np.max(new_corners[:, 1])]]).astype(np.int)


def warp_channel(image, homography):
    """
    Warps a 2D image with a given homography.
    :param image: a 2D image.
    :param homography: homograhpy.
    :return: A 2d warped image.
    """
    h = image.shape[0]
    w = image.shape[1]
    box = compute_bounding_box(homography, w, h)
    x = np.arange(box[0][0], box[1][0] + 1)
    y = np.arange(box[0][1], box[1][1] + 1)
    xv, yv = np.meshgrid(x, y)
    # stack in flip
    coordinates = np.stack((xv, yv), axis=0).T
    new_coordinates = apply_homography(coordinates.reshape(-1, 2), np.linalg.inv(homography))
    new_coordinates = np.roll(new_coordinates.reshape(coordinates.shape).T, 1, 0)
    return map_coordinates(image, new_coordinates, order=1, prefilter=False)


def warp_image(image, homography):
    """
    Warps an RGB image with a given homography.
    :param image: an RGB image.
    :param homography: homograhpy.
    :return: A warped image.
    """
    return np.dstack([warp_channel(image[..., channel], homography) for channel in range(3)])


def filter_homographies_with_translation(homographies, minimum_right_translation):
    """
    Filters rigid transformations encoded as homographies by the amount of translation from left to right.
    :param homographies: homograhpies to filter.
    :param minimum_right_translation: amount of translation below which the transformation is discarded.
    :return: filtered homographies..
    """
    translation_over_thresh = [0]
    last = homographies[0][0, -1]
    for i in range(1, len(homographies)):
        if homographies[i][0, -1] - last > minimum_right_translation:
            translation_over_thresh.append(i)
            last = homographies[i][0, -1]
    return np.array(translation_over_thresh).astype(np.int)


def estimate_rigid_transform(points1, points2, translation_only=False):
    """
    Computes rigid transforming points1 towards points2, using least squares method.
    points1[i,:] corresponds to poins2[i,:]. In every point, the first coordinate is *x*.
    :param points1: array with shape (N,2). Holds coordinates of corresponding points from image 1.
    :param points2: array with shape (N,2). Holds coordinates of corresponding points from image 2.
    :param translation_only: whether to compute translation only. False (default) to compute rotation as well.
    :return: A 3x3 array with the computed homography.
    """
    centroid1 = points1.mean(axis=0)
    centroid2 = points2.mean(axis=0)

    if translation_only:
        rotation = np.eye(2)
        translation = centroid2 - centroid1

    else:
        centered_points1 = points1 - centroid1
        centered_points2 = points2 - centroid2

        sigma = centered_points2.T @ centered_points1
        U, _, Vt = np.linalg.svd(sigma)

        rotation = U @ Vt
        translation = -rotation @ centroid1 + centroid2

    H = np.eye(3)
    H[:2, :2] = rotation
    H[:2, 2] = translation
    return H


def non_maximum_suppression(image):
    """
    Finds local maximas of an image.
    :param image: A 2D array representing an image.
    :return: A boolean array with the same shape as the input image, where True indicates local maximum.
    """
    # Find local maximas.
    neighborhood = generate_binary_structure(2, 2)
    local_max = maximum_filter(image, footprint=neighborhood) == image
    local_max[image < (image.max() * 0.1)] = False

    # Erode areas to single points.
    lbs, num = label(local_max)
    centers = center_of_mass(local_max, lbs, np.arange(num) + 1)
    centers = np.stack(centers).round().astype(np.int)
    ret = np.zeros_like(image, dtype=np.bool)
    ret[centers[:, 0], centers[:, 1]] = True

    return ret


def spread_out_corners(im, m, n, radius):
    """
    Splits the image im to m by n rectangles and uses harris_corner_detector on each.
    :param im: A 2D array representing an image.
    :param m: Vertical number of rectangles.
    :param n: Horizontal number of rectangles.
    :param radius: Minimal distance of corner points from the boundary of the image.
    :return: An array with shape (N,2), where ret[i,:] are the [x,y] coordinates of the ith corner points.
    """
    corners = [np.empty((0, 2), dtype=np.int)]
    x_bound = np.linspace(0, im.shape[1], n + 1, dtype=np.int)
    y_bound = np.linspace(0, im.shape[0], m + 1, dtype=np.int)
    for i in range(n):
        for j in range(m):
            # Use Harris detector on every sub image.
            sub_im = im[y_bound[j]:y_bound[j + 1], x_bound[i]:x_bound[i + 1]]
            sub_corners = harris_corner_detector(sub_im)
            sub_corners += np.array([x_bound[i], y_bound[j]])[np.newaxis, :]
            corners.append(sub_corners)
    corners = np.vstack(corners)
    legit = ((corners[:, 0] > radius) & (corners[:, 0] < im.shape[1] - radius) &
             (corners[:, 1] > radius) & (corners[:, 1] < im.shape[0] - radius))
    ret = corners[legit, :]
    return ret


class PanoramicVideoGenerator:
    """
    Generates panorama from a set of images.
    """

    def __init__(self, data_dir, file_prefix, num_images):
        """
        The naming convention for a sequence of images is file_prefixN.jpg,
        where N is a running number 001, 002, 003...
        :param data_dir: path to input images.
        :param file_prefix: see above.
        :param num_images: number of images to produce the panoramas with.
        """
        self.file_prefix = file_prefix
        self.files = [os.path.join(data_dir, '%s%03d.jpg' % (file_prefix, i + 1)) for i in range(num_images)]
        self.files = list(filter(os.path.exists, self.files))
        self.panoramas = None
        self.homographies = None
        print('found %d images' % len(self.files))

    def align_images(self, translation_only=False):
        """
        compute homographies between all images to a common coordinate system
        :param translation_only: see estimte_rigid_transform
        """
        # Extract feature point locations and descriptors.
        points_and_descriptors = []
        for file in self.files:
            image = sol4_utils.read_image(file, 1)
            self.h, self.w = image.shape
            pyramid, _ = sol4_utils.build_gaussian_pyramid(image, 3, 7)
            points_and_descriptors.append(find_features(pyramid))

        # Compute homographies between successive pairs of images.
        Hs = []
        for i in range(len(points_and_descriptors) - 1):
            points1, points2 = points_and_descriptors[i][0], points_and_descriptors[i + 1][0]
            desc1, desc2 = points_and_descriptors[i][1], points_and_descriptors[i + 1][1]

            # Find matching feature points.
            ind1, ind2 = match_features(desc1, desc2, .7)
            points1, points2 = points1[ind1, :], points2[ind2, :]

            # Compute homography using RANSAC.
            H12, inliers = ransac_homography(points1, points2, 100, 6, translation_only)

            # Uncomment for debugging: display inliers and outliers among matching points.
            # In the submitted code this function should be commented out!
            # display_matches(self.images[i], self.images[i+1], points1 , points2, inliers)

            Hs.append(H12)

        # Compute composite homographies from the central coordinate system.
        accumulated_homographies = accumulate_homographies(Hs, (len(Hs) - 1) // 2)
        self.homographies = np.stack(accumulated_homographies)
        self.frames_for_panoramas = filter_homographies_with_translation(self.homographies,
                                                                         minimum_right_translation=5)
        self.homographies = self.homographies[self.frames_for_panoramas]

    def generate_panoramic_images(self, number_of_panoramas):
        """
        combine slices from input images to panoramas.
        :param number_of_panoramas: how many different slices to take from each input image
        """
        assert self.homographies is not None

        # compute bounding boxes of all warped input images in the coordinate system of the middle image (as given by the homographies)
        self.bounding_boxes = np.zeros((self.frames_for_panoramas.size, 2, 2))
        for i in range(self.frames_for_panoramas.size):
            self.bounding_boxes[i] = compute_bounding_box(self.homographies[i], self.w, self.h)

        # change our reference coordinate system to the panoramas
        # all panoramas share the same coordinate system
        global_offset = np.min(self.bounding_boxes, axis=(0, 1))
        self.bounding_boxes -= global_offset

        slice_centers = np.linspace(0, self.w, number_of_panoramas + 2, endpoint=True, dtype=np.int)[1:-1]
        warped_slice_centers = np.zeros((number_of_panoramas, self.frames_for_panoramas.size))
        # every slice is a different panorama, it indicates the slices of the input images from which the panorama
        # will be concatenated
        for i in range(slice_centers.size):
            slice_center_2d = np.array([slice_centers[i], self.h // 2])[None, :]
            # homography warps the slice center to the coordinate system of the middle image
            warped_centers = [apply_homography(slice_center_2d, h) for h in self.homographies]
            # we are actually only interested in the x coordinate of each slice center in the panoramas' coordinate system
            warped_slice_centers[i] = np.array(warped_centers)[:, :, 0].squeeze() - global_offset[0]

        panorama_size = np.max(self.bounding_boxes, axis=(0, 1)).astype(np.int) + 1

        # boundary between input images in the panorama
        x_strip_boundary = ((warped_slice_centers[:, :-1] + warped_slice_centers[:, 1:]) / 2)
        x_strip_boundary = np.hstack([np.zeros((number_of_panoramas, 1)),
                                      x_strip_boundary,
                                      np.ones((number_of_panoramas, 1)) * panorama_size[0]])
        x_strip_boundary = x_strip_boundary.round().astype(np.int)

        self.panoramas = np.zeros((number_of_panoramas, panorama_size[1], panorama_size[0], 3), dtype=np.float64)
        for i, frame_index in enumerate(self.frames_for_panoramas):
            # warp every input image once, and populate all panoramas
            image = sol4_utils.read_image(self.files[frame_index], 2)
            warped_image = warp_image(image, self.homographies[i])
            x_offset, y_offset = self.bounding_boxes[i][0].astype(np.int)
            y_bottom = y_offset + warped_image.shape[0]

            for panorama_index in range(number_of_panoramas):
                # take strip of warped image and paste to current panorama
                boundaries = x_strip_boundary[panorama_index, i:i + 2]
                image_strip = warped_image[:, boundaries[0] - x_offset: boundaries[1] - x_offset]
                x_end = boundaries[0] + image_strip.shape[1]
                self.panoramas[panorama_index, y_offset:y_bottom, boundaries[0]:x_end] = image_strip

        # crop out areas not recorded from enough angles
        # assert will fail if there is overlap in field of view between the left most image and the right most image
        crop_left = int(self.bounding_boxes[0][1, 0])
        crop_right = int(self.bounding_boxes[-1][0, 0])
        assert crop_left < crop_right, 'for testing your code with a few images do not crop.'
        print(crop_left, crop_right)
        self.panoramas = self.panoramas[:, :, crop_left:crop_right, :]

    def save_panoramas_to_video(self):
        assert self.panoramas is not None
        out_folder = 'tmp_folder_for_panoramic_frames/%s' % self.file_prefix
        try:
            rmtree(out_folder)
        except:
            print('could not remove folder')
            pass
        os.makedirs(out_folder)
        # save individual panorama images to 'tmp_folder_for_panoramic_frames'
        for i, panorama in enumerate(self.panoramas):
            plt.imsave('%s/panorama%02d.png' % (out_folder, i + 1), panorama)
        if os.path.exists('%s.mp4' % self.file_prefix):
            os.remove('%s.mp4' % self.file_prefix)
        # write output video to current folder
        os.system('ffmpeg -framerate 9 -i %s/panorama%%02d.png %s.mp4' %
                  (out_folder, self.file_prefix))

    def show_panorama(self, panorama_index, figsize=(20, 20)):
        assert self.panoramas is not None
        plt.figure(figsize=figsize)
        plt.imshow(self.panoramas[panorama_index].clip(0, 1))
        plt.show()


# if __name__ == '__main__':
#     panorama = PanoramicVideoGenerator('external', 'oxford', 2)
#     panorama.align_images()
#     panorama.generate_panoramic_images(1)
#     panorama.show_panorama(0)

    # im1 = sol4_utils.read_image('external\\oxford001.jpg', sol4_utils.GREYSCALE)
    # im2 = sol4_utils.read_image('external\\oxford002.jpg', sol4_utils.GREYSCALE)
    # pyr1, _ = sol4_utils.build_gaussian_pyramid(im1, max_levels=3, filter_size=3)
    # pyr2, _ = sol4_utils.build_gaussian_pyramid(im2, max_levels=3, filter_size=3)
    # corners1, descriptors1 = find_features(pyr1)
    # corners2, descriptors2 = find_features(pyr2)
    # indices1, indices2 = match_features(descriptors1, descriptors2, min_score=0.8)
    # points1 = np.take(corners1, indices1, axis=0)
    # points2 = np.take(corners2, indices2, axis=0)
    # _, inliers_indices = ransac_homography(points1, points2, num_iter=50, inlier_tol=10)
    # display_matches(im1, im2, points1, points2, inliers_indices)

    # im = plt.imread('external\\oxford001.jpg')
    # implot = plt.imshow(im)
    # corners = spread_out_corners(sol4_utils.read_image('external\\oxford001.jpg', sol4_utils.GREYSCALE), 7, 7, 3)
    # plt.scatter(x=corners[:, 0], y=corners[:, 1], c='r', s=1)
    # plt.show()
