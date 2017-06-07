'''
Utilities
'''
import math
import numpy as np
import cv2
from config import config

def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def normalize(img):
    '''
    Normalize the grayscale image

    Arguments:
    a: the grayscale image to normalize
    Return: the normalized grayscale image
    '''
    if img.dtype != np.float32:
        img = np.array(img, dtype=np.float32)
    low = np.amin(img, axis=(0, 1))
    high = np.amax(img, axis=(0, 1))
    mid = (high + low) * 0.5
    dis = (high - low + 0.1) * 0.5  # +0.1 in case min = max
    return (img - mid) / dis

def grayscale(img):
    '''
    Convert the image to grayscale

    Arguments:
    img: the image to convert to grayscale
    Return: the converted grayscale image

    '''
    return cv2.equalizeHist(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY))

def find_houghlines(image, dest=None, rho=1, theta=np.pi/180, threshold=25, min_line_len=100, max_line_gap=40,
                    thickness=2, slop_threshold=1.963, angle=None):
    """
    image should be the output of a sobel transform.
    Returns an image with hough lines drawn.
    """
    if angle is not None:
        slop_threshold = math.cos(0.15 * np.pi) / math.sin(0.15 * np.pi)

    def fitline(lines, previous=None):
        '''
        Fit the lines with a first order polynominal
        Return: (ymin, f) where ymin is the minimal y coordinate of the lines, f is the polynominal function

        Parameters:
        lines: the lines
        previous: result of the previous fit
        '''
        if len(lines) == 0:
            return previous
        x = [a[0] for a in lines] + [a[2] for a in lines]
        y = [a[1] for a in lines] + [a[3] for a in lines]

        # weight points by their line length, penaltize relatively short lines
        ylen = [abs(a[1] - a[3]) for a in lines]
        minlen = np.min(ylen)
        maxlen = np.max(ylen)
        if maxlen - minlen < 1:
            w = None
        else:
            w = np.exp((ylen - minlen)/(maxlen - minlen) + 1.0)
            w = np.repeat(w, 2) # each weight value is for two end points

        if len(x) > 3:
            z = np.polyfit(y[:], x, 2, w=w) # weighted polynominal fit of the points
        else:
            z = np.polyfit(y[:], x, 1, w=w) # weighted polynominal fit of the points
        f = np.poly1d(z)
        return np.min(y), f

    def interpolate(lines, top, bottom, step):
        '''
        Interpolate the lines by fitting the end points with a linear polynomial function.
        bottom: specify bottom of the image, the line will extend to the bottom of the image
        the same as bottom so there is no extension.
        '''
        fits = fitline(lines)
        points = []
        if fits is None:
            return None
        for pos in range(top, bottom, step):
            points.append([int(fits[1](pos)), pos])
        return points

    def draw_lines(image, lines):
        '''
        This function draws `lines` with `color` and `thickness`.
        Lines are drawn on the image inplace (mutates the image).
        If you want to make the lines semi-transparent, think about combining
        this function with the weighted_img() function below
        '''
        if lines is None:
            return
        # filter the lines to exclude lines that are nearly horizontal
        filtered_lines = []
        for line in lines:
            for x1, y1, x2, y2 in line:
                if math.fabs(y2 - y1) > 0:
                    m = (x2 - x1) / (y2 - y1)
                    if math.fabs(m) < slop_threshold:
                        filtered_lines += [[x1, y1, x2, y2]]
                        cv2.line(image, (x1, y1), (x2, y2), 255, thickness)

    # detect_houghlines starts here
    lines = cv2.HoughLinesP(image, rho, theta, threshold, np.array([]), minLineLength=min_line_len,
                            maxLineGap=max_line_gap)
    if lines is not None:
        if dest is None:
            dest = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
        draw_lines(dest, lines)
        return dest
    else:
        return dest

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

def line_mask(image_or_shape, line, top, bottom, step, top_width, bottom_width):
    '''
    Create a line mask
    '''
    if isinstance(image_or_shape, tuple):
        image = np.zeros((bottom - top, image_or_shape[1]), dtype=np.uint8)
    else:
        image = image_or_shape
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
    