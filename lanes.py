import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from collections import deque


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
    def zeros(self, image):
        return np.zeros((image.shape[0], image.shape[1]))

    def hls_threshold(self, image, channel, thresh=(0, 255)):
        hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
        if channel == 'l':
            s_channel = hls[:, :, 1]
        else:
            s_channel = hls[:, :, 2]

        binary_output = np.zeros_like(s_channel)
        binary_output[(s_channel >= thresh[0]) & (s_channel <= thresh[1])] = 1
        return binary_output

    def gray_threshold(self, image, orient='x', thresh=(0, 255)):
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        if orient == 'x':
            sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
        else:
            sobelx = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
        abs_sobelx = np.absolute(sobelx)
        scaled_sobel = np.uint8(255.0 * abs_sobelx / np.max(abs_sobelx))

        sxbinary = np.zeros_like(scaled_sobel)
        sxbinary[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1
        return sxbinary

    def combined_threshold(self, image):
        gradx = self.gray_threshold(image, orient='x', thresh=(20, 255))
        grady = self.gray_threshold(image, orient='y', thresh=(25, 100))
        hls_s_binary = self.hls_threshold(image, 's', thresh=(120, 255))
        hls_l_binary = self.hls_threshold(image, 'l', thresh=(50, 255))

        binary = np.zeros_like(gradx)
        binary[((gradx == 1) & (grady == 1)) | ((hls_l_binary == 1) & (hls_s_binary == 1))] = 1
        return binary


class PolynomialFinder(object):
    def __init__(self):
        self.n_windows = 9
        self.margin = 100
        self.min_pixels = 50

    def find(self, image, left_line, right_line):
        self.update_fit(image, left_line) if left_line.detected else self.refind_fit(image, self.find_left_base(image),
                                                                                     left_line)
        self.update_fit(image, right_line) if right_line.detected else self.refind_fit(image,
                                                                                       self.find_right_base(image),
                                                                                       right_line)

    def find_and_draw(self, image, left_line, right_line):
        self.update_fit(image, left_line) if left_line.detected else self.refind_fit(image, self.find_left_base(image),
                                                                                     left_line)
        self.update_fit(image, right_line) if right_line.detected else self.refind_fit(image,
                                                                                       self.find_right_base(image),
                                                                                       right_line)

        out_img = np.dstack((image, image, image)) * 255
        out_img[left_line.ally, left_line.allx] = [255, 0, 0]
        out_img[right_line.ally, right_line.allx] = [0, 0, 255]

        result = cv2.addWeighted(out_img, 1, self.make_window_img(out_img, left_line.fit_y, left_line.bestx,
                                                                  right_line.bestx), 0.3, 0)
        plt.figure(figsize=(16, 8))
        plt.imshow(result)
        plt.plot(left_line.bestx, left_line.fit_y, color='yellow')
        plt.plot(right_line.bestx, right_line.fit_y, color='yellow')
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

    def update_fit(self, image, line):
        nonzero = image.nonzero()
        nonzeroy, nonzerox = np.array(nonzero[0]), np.array(nonzero[1])

        fit = line.best_fit
        lane_indexes = ((nonzerox > (fit[0] * nonzeroy ** 2 + fit[1] * nonzeroy + fit[2] - self.margin)) & (
            nonzerox < (fit[0] * nonzeroy ** 2 + fit[1] * nonzeroy + fit[2] + self.margin)))
        x = nonzerox[lane_indexes]
        y = nonzeroy[lane_indexes]

        self.update_line(line, image, x, y)

    def update_line(self, line, image, x, y):
        new_fit = np.polyfit(y, x, 2)
        ploty = np.linspace(0, image.shape[0] - 1, image.shape[0])
        fitx = new_fit[0] * ploty ** 2 + new_fit[1] * ploty + new_fit[2]
        line.update(True, new_fit, fitx, x, y, ploty)

    def refind_fit(self, image, current, line):
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

        self.update_line(line, image, x, y)

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


class Line(object):
    def __init__(self, n=5):
        self.detected = False
        self.recent_xfitted = deque([], maxlen=n)
        self.bestx = None
        self.best_fit = None
        self.current_fit = [np.array([False])]
        self.radius_of_curvature = None
        self.line_base_pos = None
        self.diffs = np.array([0, 0, 0], dtype='float')
        self.allx = None
        self.ally = None
        self.fit_y = None

    def calculate_curvature(self):
        ym_per_pixel = 30 / 720
        xm_per_pixel = 3.7 / 700

        y_eval = np.max(self.fit_y)
        fit_cr = np.polyfit(self.fit_y * ym_per_pixel, self.bestx * xm_per_pixel, 2)
        self.radius_of_curvature = ((1 + (2 * fit_cr[0] * y_eval * ym_per_pixel + fit_cr[1]) ** 2) ** 1.5) / \
                                   np.absolute(2 * fit_cr[0])

        self.line_base_pos = self.bestx[-1] * xm_per_pixel

    def update(self, detected, current_fit, fitx, x, y, fit_y):
        self.detected = detected
        self.recent_xfitted.appendleft(fitx)
        self.bestx = np.mean(self.recent_xfitted, axis=0)
        self.current_fit = current_fit
        self.allx = x
        self.ally = y
        self.fit_y = fit_y
        self.calculate_curvature()
        self.best_fit = np.polyfit(self.fit_y, self.bestx, 2)

    def last_fitx(self):
        return self.recent_xfitted[-1]


class Pipeline(object):
    def __init__(self):
        self.distortion = Distortion((9, 6))
        self.transformer = PerspectiveTransformer()
        self.binary = ThresholdBinary()
        self.finder = PolynomialFinder()
        self.left_line = Line()
        self.right_line = Line()

    def calibrate_camera(self, images):
        self.distortion.train(images)

    def do(self, image):
        undistorted = self.distortion.undistort(image)
        binary = self.binary.combined_threshold(undistorted)
        warped = self.transformer.transform(binary)
        self.finder.find(warped, self.left_line, self.right_line)

        warped_zero = np.zeros_like(warped).astype(np.uint8)
        color_warp = np.dstack((warped_zero, warped_zero, warped_zero))

        pts_left = np.array([np.transpose(np.vstack([self.left_line.bestx, self.left_line.fit_y]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([self.right_line.bestx, self.right_line.fit_y])))])
        pts = np.hstack((pts_left, pts_right))
        cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

        newwarp = self.transformer.reverse_transform(color_warp)
        result = cv2.addWeighted(undistorted, 1, newwarp, 0.3, 0)

        cv2.putText(result, 'Radius of Curvature = ' + str(
            round((self.left_line.radius_of_curvature + self.right_line.radius_of_curvature) / 2, 3)) + '(m)',
                    (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        center = (self.right_line.line_base_pos - self.left_line.line_base_pos) / 2 + self.left_line.line_base_pos
        center_left = 640 * 3.7 / 700 - center
        if center_left > 0:
            text = "{:.2f}m left of center".format(center_left)
        else:
            text = "{:.2f}m right of center".format(-center_left)
        cv2.putText(result, text, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        return result
