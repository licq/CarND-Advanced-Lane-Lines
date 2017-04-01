import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np


def draw_2_images(left_image, right_image, left_image_title=None, right_image_title=None, right_gray=False):
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    f.tight_layout()
    if left_image_title:
        ax1.set_title(left_image_title)
    ax1.imshow(left_image)
    ax1.axis('off')
    if right_image_title:
        ax2.set_title(right_image_title)
    if right_gray:
        ax2.imshow(right_image, cmap='gray')
    else:
        ax2.imshow(right_image)
    ax2.axis('off')
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0)


def draw_image(image, title=None, gray=False):
    plt.figure(figsize=(16, 8))
    if gray:
        plt.imshow(image, cmap='gray')
    else:
        plt.imshow(image)
    if title:
        plt.title(title)
    plt.axis('off')
    plt.show()


def draw_images(images, titles, columns=4):
    f, axes = plt.subplots(len(images) // columns, columns, figsize=(20, 10))
    f.tight_layout()

    for index, img in enumerate(images):
        plt.subplot(len(images) // columns, columns, index + 1)
        plt.imshow(img)
        plt.title(titles[index].split('/')[1])
        plt.axis('off')


def write_image(filename, image):
    mpimg.imsave(filename, image)


class Distortion(object):
    def __init__(self, pattern_size):
        self.pattern_size = pattern_size
        self.obj_points = []
        self.img_points = []

    def train(self, images):
        for img in images:
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            ret, corners = cv2.findChessboardCorners(gray, self.pattern_size, None)

            if ret:
                self.obj_points.append(self.obj_points_for_pattern())
                self.img_points.append(corners)

    def obj_points_for_pattern(self):
        obj_points = np.zeros((self.pattern_size[0] * self.pattern_size[1], 3), np.float32)
        obj_points[:, :2] = np.mgrid[0: self.pattern_size[0], 0: self.pattern_size[1]].T.reshape(-1, 2)
        return obj_points

    def draw_chessboard(self, image):
        copy = np.copy(image)
        gray = cv2.cvtColor(copy, cv2.COLOR_RGB2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, self.pattern_size, None)
        cv2.drawChessboardCorners(copy, self.pattern_size, corners, ret)
        return copy

    def undistort(self, image):
        image_size = (image.shape[1], image.shape[0])
        ret, mtx, dist, _, _ = cv2.calibrateCamera(self.obj_points, self.img_points, image_size, None, None)
        return cv2.undistort(image, mtx, dist, None, mtx)


class PerspectiveTransformer(object):
    def __init__(self):
        self.src = np.float32([[583, 458], [700, 458], [1030, 672], [276, 672]])

    def transform(self, image):
        image_size = (image.shape[1], image.shape[0])
        dst = np.float32([[image_size[0] * 0.2, image_size[1] * 0.1], [image_size[0] * 0.8, image_size[1] * 0.1],
                          [image_size[0] * 0.8, image_size[1] * 1], [image_size[0] * 0.2, image_size[1] * 1]])

        self.M = cv2.getPerspectiveTransform(self.src, dst)
        self.M_r = cv2.getPerspectiveTransform(dst, self.src)
        warped = cv2.warpPerspective(image, self.M, image_size, flags=cv2.INTER_LINEAR)

        return warped

    def reverse_transform(self, image):
        return cv2.warpPerspective(image, self.M_r, (image.shape[1], image.shape[0]))


class ThresholdBinary(object):
    def __init__(self, sobelx_thresh=(20, 100), saturation_thresh=(170, 255)):
        self.sobelx_thresh = sobelx_thresh
        self.saturation_thresh = saturation_thresh

    def zeros(self, image):
        return np.zeros((image.shape[0], image.shape[1]))

    def saturation_threshold(self, image):
        hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
        s_channel = hls[:, :, 2]

        binary_output = np.zeros_like(s_channel)
        binary_output[(s_channel >= self.saturation_thresh[0]) & (s_channel <= self.saturation_thresh[1])] = 1
        return binary_output

    def gray_threshold(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
        abs_sobelx = np.absolute(sobelx)
        scaled_sobel = np.uint8(255.0 * abs_sobelx / np.max(abs_sobelx))

        sxbinary = np.zeros_like(scaled_sobel)
        sxbinary[(scaled_sobel >= self.sobelx_thresh[0]) & (scaled_sobel <= self.sobelx_thresh[1])] = 1
        return sxbinary

    def combined_threshold(self, image):
        s_channel_binary = self.saturation_threshold(image)
        sobelx_binary = self.gray_threshold(image)

        binary = np.zeros_like(s_channel_binary)
        binary[(s_channel_binary == 1) | (sobelx_binary == 1)] = 1
        return binary


class PolynomialFinder(object):
    def __init__(self):
        self.n_windows = 9
        self.margin = 100
        self.min_pixels = 50
        self.left_fit = None
        self.right_fit = None

    def find(self, image):
        self.left_fit, _ = self.update_fit(self.left_fit, image) if self.left_fit is not None \
            else self.refind_fit(image, self.find_left_base(image))
        self.right_fit, _ = self.update_fit(self.right_fit, image) if self.right_fit is not None \
            else self.refind_fit(image, self.find_right_base(image))

        return self.left_fit, self.right_fit

    def find_and_draw(self, image):
        self.left_fit, left_lane_indexes = self.update_fit(self.left_fit, image) if self.left_fit is not None \
            else self.refind_fit(image, self.find_left_base(image))
        self.right_fit, right_lane_indexes = self.update_fit(self.right_fit, image) if self.right_fit is not None \
            else self.refind_fit(image, self.find_right_base(image))

        nonzero = image.nonzero()
        nonzeroy, nonzerox = np.array(nonzero[0]), np.array(nonzero[1])

        out_img = np.dstack((image, image, image)) * 255
        out_img[nonzeroy[left_lane_indexes], nonzerox[left_lane_indexes]] = [255, 0, 0]
        out_img[nonzeroy[right_lane_indexes], nonzerox[right_lane_indexes]] = [0, 0, 255]

        ploty = np.linspace(0, image.shape[0] - 1, image.shape[0])
        left_fitx = self.left_fit[0] * ploty ** 2 + self.left_fit[1] * ploty + self.left_fit[2]
        right_fitx = self.right_fit[0] * ploty ** 2 + self.right_fit[1] * ploty + self.right_fit[2]
        result = cv2.addWeighted(out_img, 1, self.make_window_img(out_img, ploty, left_fitx, right_fitx), 0.3, 0)
        plt.figure(figsize=(16, 8))
        plt.imshow(result)
        plt.plot(left_fitx, ploty, color='yellow')
        plt.plot(right_fitx, ploty, color='yellow')
        plt.xlim(0, 1280)
        plt.ylim(720, 0)

    def make_window_img(self, image, ploty, left_fitx, right_fitx):
        window_img = np.zeros_like(image)
        left_line_window1 = np.array([np.transpose(np.vstack([left_fitx - self.margin, ploty]))])
        left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx + self.margin, ploty])))])
        left_line_pts = np.hstack((left_line_window1, left_line_window2))

        right_line_window1 = np.array([np.transpose(np.vstack([right_fitx - self.margin, ploty]))])
        right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx + self.margin, ploty])))])
        right_line_pts = np.hstack((right_line_window1, right_line_window2))

        cv2.fillPoly(window_img, np.int_([left_line_pts]), (0, 255, 0))
        cv2.fillPoly(window_img, np.int_([right_line_pts]), (0, 255, 0))

        return window_img

    def update_fit(self, fit, image):
        nonzero = image.nonzero()
        nonzeroy, nonzerox = np.array(nonzero[0]), np.array(nonzero[1])

        lane_indexes = ((nonzerox > (fit[0] * nonzeroy ** 2 + fit[1] * nonzeroy + fit[2] - self.margin)) & (
            nonzerox < (fit[0] * nonzeroy ** 2 + fit[1] * nonzeroy + fit[2] + self.margin)))
        x = nonzerox[lane_indexes]
        y = nonzeroy[lane_indexes]

        return np.polyfit(y, x, 2), lane_indexes

    def refind_fit(self, image, current):
        nonzero = image.nonzero()
        nonzeroy, nonzerox = np.array(nonzero[0]), np.array(nonzero[1])

        w_height = np.int(image.shape[0] / self.n_windows)
        lane_indexes = []

        for w in range(self.n_windows):
            win_y_low, win_y_high = image.shape[0] - (w + 1) * w_height, image.shape[0] - w * w_height
            win_x_low, win_x_high = current - self.margin, current + self.margin

            good_indexes = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_x_low) & (
                nonzerox < win_x_high)).nonzero()[0]
            lane_indexes.append(good_indexes)

            if len(good_indexes) >= self.min_pixels:
                current = np.int(np.mean(nonzerox[good_indexes]))

        lane_indexes = np.concatenate(lane_indexes)
        x = nonzerox[lane_indexes]
        y = nonzeroy[lane_indexes]

        return np.polyfit(y, x, 2), lane_indexes

    @staticmethod
    def find_left_base(image):
        histogram = np.sum(image[np.int(image.shape[0] / 2):, :], axis=0)
        midpoint = np.int(histogram.shape[0] / 2)

        left_base = np.argmax(histogram[:midpoint])
        return left_base

    @staticmethod
    def find_right_base(image):
        histogram = np.sum(image[np.int(image.shape[0] / 2):, :], axis=0)
        midpoint = np.int(histogram.shape[0] / 2)

        right_base = np.argmax(histogram[midpoint:]) + midpoint
        return right_base


class Pipeline(object):
    def __init__(self):
        self.distortion = Distortion((9, 6))
        self.transformer = PerspectiveTransformer()
        self.binary = ThresholdBinary()
        self.finder = PolynomialFinder()
        self.ym_per_pixel = 30 / 720
        self.xm_per_pixel = 3.7 / 700

    def calibrate_camera(self, images):
        self.distortion.train(images)

    def do(self, image):
        undistorted = self.distortion.undistort(image)
        binary = self.binary.combined_threshold(undistorted)
        warped = self.transformer.transform(binary)
        left_fit, right_fit = self.finder.find(warped)

        ploty = np.linspace(0, image.shape[0] - 1, image.shape[0])
        left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
        right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

        warped_zero = np.zeros_like(warped).astype(np.uint8)
        color_warp = np.dstack((warped_zero, warped_zero, warped_zero))

        pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
        pts = np.hstack((pts_left, pts_right))
        cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

        newwarp = self.transformer.reverse_transform(color_warp)
        result = cv2.addWeighted(undistorted, 1, newwarp, 0.3, 0)

        y_eval = np.max(ploty)
        left_fit_cr = np.polyfit(ploty * self.ym_per_pixel, left_fitx * self.xm_per_pixel, 2)
        right_fit_cr = np.polyfit(ploty * self.ym_per_pixel, right_fitx * self.xm_per_pixel, 2)
        left_curverad = ((1 + (
            2 * left_fit_cr[0] * y_eval * self.ym_per_pixel + left_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
            2 * left_fit_cr[0])
        right_curverad = ((1 + (
            2 * right_fit_cr[0] * y_eval * self.ym_per_pixel + right_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
            2 * right_fit_cr[0])
        cv2.putText(result, 'Radius of Curvature = ' + str(round((left_curverad + right_curverad) / 2, 3)) + '(m)',
                    (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        return result
