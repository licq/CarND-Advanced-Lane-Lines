import cv2
import matplotlib.pyplot as plt
import numpy as np


def warper(img, src, dst):
    # Compute and apply perpective transform
    img_size = (img.shape[1], img.shape[0])
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_NEAREST)  # keep same size as input image

    return warped


def draw_2_images(left_image, right_image, left_image_title=None, right_image_title=None):
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    f.tight_layout()
    if left_image_title:
        ax1.set_title(left_image_title)
    ax1.imshow(left_image)
    if right_image_title:
        ax2.set_title(right_image_title)
    ax2.imshow(right_image)
    plt.axis('off')
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0)


def draw_images(images, titles, columns=4):
    f, axes = plt.subplots(len(images) // columns, columns, figsize=(20, 10))
    f.tight_layout()

    for index, img in enumerate(images):
        plt.subplot(len(images) // columns, columns, index + 1)
        plt.imshow(img)
        plt.title(titles[index].split('/')[1])
        plt.axis('off')


def write_image(filename, image):
    cv2.imwrite(filename, image)


class Distortion(object):
    def __init__(self, pattern_size):
        self.pattern_size = pattern_size
        self.obj_points = []
        self.img_points = []

    def train(self, images):
        for img in images:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
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
        gray = cv2.cvtColor(copy, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, self.pattern_size, None)
        cv2.drawChessboardCorners(copy, self.pattern_size, corners, ret)
        return copy

    def undistort(self, image):
        image_size = (image.shape[1], image.shape[0])
        ret, mtx, dist, _, _ = cv2.calibrateCamera(self.obj_points, self.img_points, image_size, None, None)
        return cv2.undistort(image, mtx, dist, None, mtx)
