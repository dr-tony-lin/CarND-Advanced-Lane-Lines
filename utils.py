import numpy as np
import cv2
from config import config

def draw_lane(shape, top, bottom, step, left, right):
    # Create an image to draw the lines on
    image = np.zeros(shape, dtype=np.uint8)
    left_points = []
    right_points = []
    for i in range(top, bottom, step):
        if config.bestfit and left.best_fit is not None:
            left_points.append([int(left.bestx(i)), i])
        else:
            left_points.append([int(left.x(i)), i])
        if config.bestfit and right.best_fit is not None:
            right_points.append([int(right.bestx(i)), i])
        else:
            right_points.append([int(right.x(i)), i])
    pts = np.vstack((np.array(left_points), np.array(right_points)[::-1]))
    if config.test:
        print(left_points)
        print(right_points)
        print(pts)

    # Draw the lane onto the warped blank image
    cv2.fillPoly(image, np.int_([pts]), (0, 255, 0))
    return image

def line_mask(shape, line, top, bottom, step, top_width, bottom_width):
    '''
    Create a line mask
    '''
    image = np.zeros((bottom - top, shape[1]), dtype=np.uint8)
    left_points = []
    right_points = []
    grad = (bottom_width - top_width) / (bottom - top)
    for i in range(top, bottom, step):
        if config.bestfit:
            x = int(line.bestx(i))
        else:
            x = int(line.x(i))
        w = grad * (i - top) + top_width
        left_points.append([int(x-w), i - top])
        right_points.append([int(x+w), i - top])

    pts = np.vstack((np.array(left_points), np.array(right_points)[::-1]))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(image, np.int_([pts]), 255)
    return image

def weighted_img(image, overlay, α=0.8, β=0.2, λ=0.):
    """
    `overlay` is the overlay
    `image` should be the image before any processing.
    The result image is computed as follows:
    image * α + overlay * β + λ
    NOTE: image and ioverlaymg must be the same shape!
    """
    return cv2.addWeighted(image, α, overlay, β, λ)