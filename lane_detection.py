'''
Lane detetion
'''
import glob
import math
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

class Camera:
    '''
    Provide camera calibration and transformation
    '''
    def __init__(self):
        # mtx of the camera
        self.mtx = None
        # dist of the camera
        self.dist = None
        # Perspective transformation matrix to birds eye
        self.trans = None
        # Inverse perspective transformation matrix to the projection space
        self.invtrans = None
        # Height of the birds eye image to transform to
        self.dest_height = 0

    def set_transformation(self, dest_height=200, x=[275, 565, 720, 1045], y=[470, 680]):
        '''
        Set perspective transformation
        Arguments:
        dest_height: height of the destination image
        x: the x coordinates of the source trapezoid that will be transformed to a rectangle by the transformation
        y: the top and base y coordinate of the source trapezoid
        '''
        self.dest_height = dest_height
        imgpoints = np.float32([[x[1], y[0]], [x[2], y[0]], [x[3], y[1]], [x[0], y[1]]])
        objpoints = np.float32([[imgpoints[3][0], 0], [imgpoints[2][0], 0],
                                [imgpoints[2][0], dest_height], [imgpoints[3][0], dest_height]])
        self.trans = cv2.getPerspectiveTransform(imgpoints, objpoints)
        self.invtrans = cv2.getPerspectiveTransform(objpoints, imgpoints)
        return imgpoints, objpoints

    def calibrate(self, images, chessboard_shape):
        '''
        Calibrate camera
        Arguments:
        images: chessboard calibration images folder and file name pattern, e.g. camera_cal/*.jpg
        chessboard_shape: shape (number grids) of the chessboard
        '''
        objpoints = []
        imgpoints = []
        objp = np.zeros((chessboard_shape[0]*chessboard_shape[1], 3), dtype=np.float32)
        objp[:, :2] = np.mgrid[0:chessboard_shape[0], 0:chessboard_shape[1]].T.reshape(-1, 2)
        for name in glob.glob(images):
            image = cv2.cvtColor(mpimg.imread(name), cv2.COLOR_RGB2GRAY)
            ret, corners = cv2.findChessboardCorners(image, chessboard_shape, None)
            if ret:
                objpoints.append(objp)
                imgpoints.append(corners)
        ret, mtx, dist, _, _ = cv2.calibrateCamera(objpoints, imgpoints, image.shape[::-1], None, None)
        if ret:
            self.mtx = mtx
            self.dist = dist
        return ret, mtx, dist

    def undistort(self, image):
        '''
        Un-distort the image
        image: the image to un-distort
        '''
        return cv2.undistort(image, self.mtx, self.dist, None, self.mtx)

    def parallel(self, image):
        '''
        Perform parallel transformion on the image
        image: the image to transform
        '''
        image = cv2.warpPerspective(image, self.trans, (image.shape[1], self.dest_height), flags=cv2.INTER_LINEAR)
        return image

    def perspective(self, points):
        '''
        Perform perspective transform for the points
        points: the points to transform
        '''
        transformed = cv2.perspectiveTransform(np.float32([points]), self.invtrans)
        return transformed[0]

class Lane():
    def __init__(self):
        # was the line detected in the last iteration?
        self.detected = False
        # Number of consecutive fail detections
        self.fails = 0
        # x values of the last n detections
        self.recent_detects = []
        # x values of the last n fits of the line
        self.recent_fits = []
        #polynomial coefficients averaged over the last n iterations
        self.best_fit = None
        #polynomial coefficients for the most recent fit
        self.current_fit = None
        # The last detected lane points
        self.current = None
        #radius of curvature of the line in some units
        self.radius_of_curvature = None
        #distance in meters of vehicle center from the line
        self.line_base_pos = None
        self.history_size = 3

    def set(self, points):
        '''
        Set the fit's coefficient
        '''
        self.current = points
        if points is not None and len(points) > 1: # we need at least two points
            self.detected = True
            self.fails = 0
            self.recent_detects.append(points)
            if len(self.recent_detects) > self.history_size:
                self.recent_detects.pop(0)
            # give more weight to the bottom of the lone
            w = np.ones((len(points), ), np.float)
            w[0] = 2.0
            if len(points) > 3: # minimal 4 points for second order polynomial
                self.current_fit = np.polyfit(points[:, 1], points[:, 0], 2, w=w)
            else: # first order polynomial, coefficient 0 (the square term) is 0
                self.current_fit = np.insert(np.polyfit(points[:, 1], points[:, 0], 1, w=w), 0, 0.)

            self.recent_fits.append(self.current_fit)
            if len(self.current_fit) > self.history_size:
                self.current_fit.pop(0)

            if self.best_fit is None:
                self.best_fit = self.current_fit
            else:
                # Should we use recent_detects to compute the best fit?
                self.best_fit = self.best_fit * 0.4 + self.current_fit * 0.6
        else:
            self.detected = False
            self.fails += 1

    def curverature(self, y, poly):
        return math.pow((1. + (2*poly[0]*y + poly[1])**2), 3./2.) / abs(2*poly[0])

    def x(self, y):
        '''
        Return the x coordinate at y with the current fit
        '''
        return np.polyval(self.current_fit, y)

    def bestx(self, y):
        '''
        Return the x coordinate at y with the current fit
        '''
        return np.polyval(self.best_fit, y)

class LaneDetector:
    '''
    Lane detector class that use an image processing pipeline to detect lanes
    '''
    def __init__(self, config):
        '''
        camers: the camera used for capturing the images. It must has been calibrated
        config: the configuration
        '''
        # Camera
        self.camera = Camera()
        # Detection configuration
        self.config = config
        config.crop = sorted(config.crop)
        config.proj_x = sorted(config.proj_x)
        # The ytop and bottom coordinates of the trapezoid
        config.proj_y = sorted(config.proj_y)
        config.sobel_thresh = sorted(config.sobel_thresh)
        config.magnitude_thresh = sorted(config.magnitude_thresh)
        # Normalized sobel threshold
        self.sobel_thresh = np.float32(config.sobel_thresh)/255.0
        # Normalized sobel mangitude threshold
        self.magnitude_thresh = np.float32(config.magnitude_thresh)/255.0
        # Normalized HLS threshold
        if config.hls_thresh is None:
            self.hls_thresh = None
        else:
            config.hls_thresh = sorted(config.hls_thresh)
            self.hls_thresh = np.float32(config.hls_thresh)/255.0

        if config.crop is not None:
            self.image_size = (config.crop[1] - config.crop[0], config.image_size[1])
            if config.proj_y is not None:
                self.proj_y = [config.proj_y[0] - config.crop[0], config.proj_y[1] - config.crop[0]]
            else:
                self.proj_y = [config.proj_y[0] - config.crop[1], config.proj_y[1] - config.crop[1]]
        else:
            self.image_size = config.image_size
            self.proj_y = config.proj_y

        print("Calibrating camera ...")
        ret, mtx, dist = self.camera.calibrate(config.calibration_folder + "*.jpg",
                                               chessboard_shape=config.chessboard_shape)
        assert ret, "Failed to calibrate camera!"

        if self.config.test:
            print("Calibrated undistortion params: ", mtx, dist)

        imgpoints, objpoints = self.camera.set_transformation(self.config.layer_height*self.config.scan_layers,
                                                              config.proj_x, self.proj_y)
        if self.config.test:
            print("Detection image height: ", self.config.layer_height*self.config.scan_layers)
            print("Scan thresh", self.config.scan_thresh)
            print("Image points:")
            print(imgpoints)
            print("Object points:")
            print(objpoints)
            print("Transformation matrix:")
            print(self.camera.trans)
            print("Inverse transformation matrix:")
            print(self.camera.invtrans)

        # Left lane detected
        self.left_lane = Lane()
        # Right lane detected
        self.right_lane = Lane()
        # Convolution window
        self.convolution = np.ones(self.config.sliding_width)

    def apply_hsv_thresh(self, image):
        '''
        Apply threshold of the white and yellow colors of the HSV colorspace
        image: the image
        '''
        if self.config.hsv_thresh is None:
            return None
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
        for (lower, upper) in self.config.hsv_thresh: # filter range
            mask |= cv2.inRange(hsv, lower, upper)
        return mask

    def apply_hls_thresh(self, image):
        '''
        Apply the threshold of the saturation of the HLS colorspace
        '''
        if self.hls_thresh is None:
            return None
        hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
        channel = hls[:, :, 2]
        max_saturation = np.amax(channel)
        hls_thresh = self.hls_thresh * max_saturation
        return (channel >= hls_thresh[0]) & (channel <= hls_thresh[1])

    def apply_threshold(self, image):
        '''
        Apply threshold to the image for extracting lane feature
        image: the image
        Return: an image with lane features extracted
        '''
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        sobelx = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=self.config.sobel_kernel))
        sobely = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=self.config.sobel_kernel))
        sobelm = np.sqrt(sobelx * sobelx + sobely * sobely)
        sobelx = sobelx/(np.amax(sobelx)+1e-6)
        sobely = sobely/(np.amax(sobely)+1e-6)
        sobelm = sobelm/(np.amax(sobelm)+1e-6)
        extracted = np.zeros_like(gray, dtype=np.uint8)
        hls = self.apply_hls_thresh(image)
        hsv = self.apply_hsv_thresh(image)
        passed = (((sobelx >= self.sobel_thresh[0]) & (sobelx <= self.sobel_thresh[1]) &
                   (sobely >= self.sobel_thresh[0]) & (sobely <= self.sobel_thresh[1])) |
                  ((sobelm >= self.magnitude_thresh[0]) & (sobelm <= self.magnitude_thresh[1])))
        if hls is not None:
            passed = passed | hls
        if hsv is not None:
            passed = passed | (hsv > 0)
        extracted[passed] = 1
        if self.config.test:
            hls_out = None
            if hls is not None:
                hls_out = np.zeros_like(hls, dtype=np.uint8)
                hls_out[hls] = 1
            return extracted, sobelx, sobely, sobelm, hls_out, hsv
        return extracted, None, None, None, None, None

    def preprocess(self, image):
        '''
        Proprecess the image for lane detection. It will un-distort the image, then extract lanes features,
        then apply reverse perspective transformation
        images: the image
        Return: an image ready for finding lanes
        '''
        undistorted = self.camera.undistort(image)
        image = undistorted[self.config.crop[0]:self.config.crop[1], :, :]
        extracted, sobelx, sobely, sobelm, hls, hsv = self.apply_threshold(image)
        tranformed = self.camera.parallel(extracted)
        if self.config.test:
            return undistorted, tranformed, image, extracted, sobelx, sobely, sobelm, hls, hsv
        else:
            return undistorted, tranformed, None, None, None, None, None, None, None

    def draw_window(self, image, x, y, level, box_color=[0, 255, 0], color=[255, 0, 0]):
        x = int(x)
        y = int(y)
        half = int(self.config.sliding_width/2)
        ybot = int(image.shape[0] - level * self.config.layer_height)
        ytop = int(ybot - self.config.layer_height)
        image[ytop:ybot, max(0, x-half):min(x+half, image.shape[1]), :] = box_color
        cv2.circle(image, (x, y), 8, color, -1)

    def reset(self):
        self.left_lane = Lane()
        self.right_lane = Lane()
        
    def start_detect(self, image):
        '''
        Find the starting point for lane detection
        image: the image
        '''
        # First find the two starting positions for the left and right lane by using np.sum to get the
        # vertical image slice and then np.convolve the vertical image slice with the window template
        # Sum half bottom of image to get slice, could use a different ratio
        half = self.config.sliding_width/2
        left_hist = np.sum(image[int(image.shape[0]/2):, :int(image.shape[1]/2)], axis=0)
        leftx = np.argmax(np.convolve(self.convolution, left_hist)) - half
        right_hist = np.sum(image[int(image.shape[0]/2):, int(image.shape[1]/2):], axis=0)
        rightx = np.argmax(np.convolve(self.convolution, right_hist)) - half + int(image.shape[1]/2)
        if self.config.test:
            print("Start at: ", leftx, rightx)
        return int(leftx), int(rightx)

    def scan(self, image, previous=None):
        '''
        Detect lane lines by scanning the binary image
        Arguments:
        image: the binary image
        previous: x coordinate of the previous starting point (left, right)
        '''
        self.config.layer_height = image.shape[0] / self.config.scan_layers
        left_lane = [] # Store the left lane points
        right_lane = [] # Store the right lane points
        half = int(self.config.sliding_width/2)

        if previous is None:
            leftx, rightx = self.start_detect(image)
        else:
            leftx, rightx = previous[0], previous[1]

        if self.config.test: # Create visualization image
            visual = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)
        else:
            visual = None

        # Go through each layer looking for max pixel locations
        for i in range(0, self.config.scan_layers):
            # convolve the window into the vertical slice of the image
            ybot = int(image.shape[0] - i * self.config.layer_height)
            ytop = int(ybot - self.config.layer_height)
            histogram = np.sum(image[ytop:ybot, :], axis=0)
            convolution = np.convolve(self.convolution, histogram)
            # Find the best left centroid by using past left center as a reference
            # Use self.config.sliding_width/2 as offset because convolution signal reference is at
            # right side of window, not center of window
            left_min = int(max(leftx + half - self.config.scan_width, 0))
            left_max = int(min(leftx + half + self.config.scan_width, image.shape[1]))
            right_min = int(max(rightx + half - self.config.scan_width, 0))
            right_max = int(min(rightx + half + self.config.scan_width, image.shape[1]))

            # argmax will not return valid index if no pixel is on, or there are noise
            # we will set leftx, rightx when the convolution exceed the threshold
            leftx_tmp = np.argmax(convolution[left_min:left_max]) + left_min - half
            rightx_tmp = np.argmax(convolution[right_min:right_max]) + right_min - half

            # TODO: try if using average of on points to compute x and y can yield better fits
            ymid = int(ytop + self.config.layer_height/2)
            if convolution[leftx_tmp] >= self.config.scan_thresh:
                # found left lane for that layer, accept the point and set the new leftx
                leftx = leftx_tmp
                left_lane.append([leftx, ymid])
                if self.config.test: # Draw visualization image
                    self.draw_window(visual, leftx, ymid, i)
            elif i == 0: # the first layer, but we could find y, use the one from start
                left_lane.append([leftx, ymid])
                if self.config.test: # Draw visualization image
                    self.draw_window(visual, leftx, ymid, i)
            elif self.config.test:
                print("Skip left: ", leftx, ymid, left_min, left_max, convolution[leftx_tmp])

            if convolution[rightx_tmp] >= self.config.scan_thresh: # found right lane for that layer
                # found right lane for that layer, accept the point and set the new leftx
                rightx = rightx_tmp
                right_lane.append([rightx, ymid])
                if self.config.test: # Draw visualization image
                    self.draw_window(visual, rightx, ymid, i)
            elif i == 0: # the first layer, but we could find y, use the one from start
                right_lane.append([rightx, ymid])
                if self.config.test: # Draw visualization image
                    self.draw_window(visual, rightx, ymid, i)
            elif self.config.test:
                print("Skip right: ", rightx, ymid, right_min, right_max, convolution[rightx_tmp])

            if self.config.test:
                print("Layer center: ", leftx, rightx)

        if len(left_lane) > 0: # we have enough point for a polyline
            left_lane = self.camera.perspective(left_lane) 
            left_lane[:, 1] = left_lane[:, 1] + self.config.crop[0]
            self.left_lane.set(left_lane)
            if self.config.test:
                print("Transform left lane: ", left_lane)
                print("Left lane polynomial: ", self.left_lane.current_fit)
        else:
            self.left_lane.set(None)
        if len(right_lane) > 0: # we have enough point for a polyline
            right_lane = self.camera.perspective(right_lane)
            right_lane[:, 1] = right_lane[:, 1] + self.config.crop[0]
            self.right_lane.set(right_lane)
            if self.config.test:
                print("Transform right lane: ", right_lane)
                print("Right lane polynomial: ", self.right_lane.current_fit)
        else:
            self.left_lane.set(None)

        return visual

    def _previous_start(self):
        return None
        if (self.left_lane.current is not None) and (self.right_lane.current is not None):
            if ((self.image_size[0] - self.left_lane.current[0][1]) < self.config.layer_height) and \
               ((self.image_size[0] - self.right_lane.current[0][1]) < self.config.layer_height):
                return self.left_lane.current[0][0], self.right_lane.current[0][0]

    def detect(self, image):
        '''
        Detect lane
        image: the image to detect
        Return: the left and right lanes in format [[Lx0, Lx1, Lx2], [Rx0, Rx1, Rx2]]
        '''
        undistorted, trasnsformed, undistort, extracted, sobelx, sobely, sobelm, hls, hsv = self.preprocess(image)
        previous_start = self._previous_start()
        visual = self.scan(trasnsformed, previous_start)
        if self.config.test:
            return undistorted, visual, trasnsformed, undistort, extracted, sobelx, sobely, sobelm, hls, hsv
        else:
            return undistorted, None, None, None, None, None, None, None, None, None
        