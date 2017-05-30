'''
Configuration for Advanced Lane Detection
'''
import math
import numpy as np

class Config:
    pass

config = Config()

# Test flag, True to allow more data to be made available for test
config.test = True
# Use best fit to fit lanes
config.bestfit = True
# True to save video images
config.save_video_images = False
# Folder containing the camera calibration images
config.calibration_folder = "camera_cal/"
# Folder containing the test images
config.test_image_folder = "test_images/"
# Size of the camera images
config.image_size = (720, 1280)
# Chessboard cells' shape
config.chessboard_shape = (9,6)
# The horizontal area of the camera image to be used for lane detection
config.crop=(470, 690)
# The x coordinates of the trapezoid on the camera image for computing perspective transformation matrix
config.proj_x = [260, 565, 720, 1060]
# The ytop and bottom coordinates of the trapezoid before cropping
config.proj_y = [470, 690]
# The number of vertical layers on the transformed image for used for lane detection
config.scan_layers = 8
# Height of a scan layer
config.layer_height = 90#int(math.ceil((config.crop[1] - config.crop[0]) / config.scan_layers))
# Width of the sliding windows in each scan
config.sliding_width = 50
# Width of the scan range to the left and right of the lane detected in the layer below
config.scan_width = 100
# The number of on pixels in a sliding window that need to be on for the window to be considered 
config.scan_thresh = int(config.sliding_width * config.layer_height * 0.15)
# Sobel detection kernel
config.sobel_kernel = 3
# Sobel threshold
config.sobel_thresh = (20, 100)
# Sobel magnitude threshold 
config.magnitude_thresh = (80, 180)
# HSL colorspace satuation threshold
config.hls_thresh = (160, 225)
# HSV threshold to filter image by white and yellow colors. The first range is for pure white,
# the second is for near white which can be any color with low saturation, and the third is for yellow
config.hsv_thresh = [(np.uint8([0, 0, 120]), np.uint8([0, 0, 255])),
                     (np.uint8([0, 0, 220]), np.uint8([180, 20, 255])),
                     (np.uint8([18, 80, 120]), np.uint8([25, 255, 255]))]
