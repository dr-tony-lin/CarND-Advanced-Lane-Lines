'''
Configuration for Advanced Lane Detection
'''
import os
import numpy as np

class Config:
    '''
    Config class
    '''
    def __init__(self):
        self.crop = None
        self.trapezoid_y = None
        self.trapezoid_x = None
        self.scan_thresh = None

    def set(self, name=None):
        '''
        Set configuration
        name: name of the configuration
        '''
        trapezoids = None
        if name is not None:
            name = os.path.splitext(os.path.basename(name))[0]
            for key in self.trapezoid:
                if name.find(key) == 0:
                    print('Use configuration: ', key)
                    trapezoids = self.trapezoid[key]
                    break
        if trapezoids is None:
            print('Use default configuration')
            trapezoids = self.trapezoid['default']
        self.crop = trapezoids[0]
        self.trapezoid_y = trapezoids[0]
        self.trapezoid_x = trapezoids[1]
        self.scan_thresh = int(self.sliding_width * self.layer_height * self.scan_thresh_ratio)

config = Config()

# Test flag, True to allow more data to be made available for test
config.test = True
# Use best fit to fit lanes
config.bestfit = True
# Use best fit to fit lanes
config.lane_shift_thresh = 75
# True to save video images
config.save_video_images = False
# Parallel trapezoids for computing parallel-perspective projection transformation for different videos
config.trapezoid = {
    'default': [[470, 690], [260, 565, 720, 1060]],
    'challenge_video': [[490, 690], [332, 590, 740, 1080]],
    'harder_challenge_video': [[500, 675], [250, 500, 737, 963]]
}
# Folder containing the camera calibration images
config.calibration_folder = "camera_cal/"
# Folder containing the test images
config.test_image_folder = "test_images/"
# Size of the camera images
config.image_size = (720, 1280)
# Chessboard cells' shape
config.chessboard_shape = (9,6)
# The number of vertical layers on the transformed image for used for lane detection
config.scan_layers = 16
# Height of a scan layer
config.layer_height = 90
# Width of the sliding windows in each scan
config.sliding_width = 50
# Width of the scan range to the left and right of the lane detected in the layer below
config.scan_width = 100
# The number of on pixels in a sliding window that need to be on for the window to be considered 
config.scan_thresh_ratio = 0.15
# Sobel detection kernel
config.sobel_kernel = 3
# Sobel threshold
config.sobel_thresh = (20, 100)
# Sobel magnitude threshold 
config.magnitude_thresh = (80, 180)
# HSL colorspace satuation threshold
config.hls_thresh = None #(160, 225)
# HSV threshold to filter image by white and yellow colors. The first range is for pure white,
# the second is for near white which can be any color with low saturation, and the third is for yellow
config.hsv_thresh = [(np.uint8([0, 0, 215]), np.uint8([180, 30, 255])),
                     (np.uint8([18, 80, 120]), np.uint8([25, 255, 255]))]

# Lane change mark color
config.hsv_maskoff = [(np.uint8([0, 0, 0]), np.uint8([180, 30, 120]))]

config.set()
