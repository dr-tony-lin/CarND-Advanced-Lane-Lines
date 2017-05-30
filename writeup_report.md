# Advanced Lane Finding Project

## Goals

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1-1]: ./camera_cal/calibration1.jpg "Calibration1"
[image1-2]: ./examples/calibration1-2.jpg "Calibration1 Undistorted"
[image2-1]: ./camera_cal/calibration2.jpg "Calibration2"
[image2-2]: ./examples/calibration2-2.jpg "Calibration2 Undistorted"
[image3-1]: ./camera_cal/calibration3.jpg "Calibration3"
[image3-2]: ./examples/calibration3-2.jpg "Calibration3 Undistorted"
[image4]: ./test_images/test5.jpg "Road Original"
[image4-1]: ./examples/test5-undistort.jpg "Undistorted"
[image5-1]: ./examples/test5-bin.jpg "Binary Example"
[image5-2]: ./examples/test5-sobelx.jpg "Sobelx Example"
[image5-3]: ./examples/test5-sobely.jpg "Sobely Example"
[image5-4]: ./examples/test5-sobelm.jpg "Sobel Mangitude Example"
[image5-5]: ./examples/test5-hls.jpg "HLS Example"
[image5-6]: ./examples/test5-hsv.jpg "HSV Example"
[image5-7]: ./examples/test5-lines.jpg "Warp Example"
[image5-8]: ./examples/test5-trace.jpg "Fit Visual"
[image6]: ./examples/test5-final.jpg "Output"
[video1]: ./project_video.mp4 "Video"

## Python Files
The following python files are included in this submission:

* run.py: this python drive lane detection process for all mp4 video in this folder
* config.py: contains lane detection configurations
* lane_detection.py: contains the following classes: 
  * Camera: for camera correction and transformation
  * Lane: implement lane that keeps track of lane detection in a video
  * LaneDetector: provide the lane detection functionality
* test.py: a program used to test camera calibration, thresholding, transformation, and detection
* utils.py: contains some utility functions
* show_channels.py: a simple program to show color channels

### Usage
The run.py program does not need any argument, simply run:
```
python run.py
```

## Camera Calibration

Camera calibration was performed by *Camera.calibrate()* which goes through all chessboard images in *camera_cal* folder. 

For each image, four chessboard corner points in 3D space, (x1, y1, 0), (x2, y2, 0), (x3, y3, 0), (x4, y4, 0) are computed from the chessboard's gemoetry. 

Then the corresponding 4 point in the image are detected using *cv2.findChessboardCorners()* function. After every image is processed, *cv2.calibrateCamera()* are invoked with the points to obtain mtx, and dist parameters.

The mtx, and dist parameters are then used in *cv2.undistort() to undistort camera images.

The following diagrams show some of the chessboard images and their undistorted counterparts.

|           		    |   Examples of distorted Images  |   	    |
|:-----------------:|:-------------------:|:-------------------:| 
| ![image1-1]       | ![image2-1]  		    | ![image3-1]         |
|           		|   Their Uundistorted Images  |   	            |
| ![image1-2]       | ![image2-2]  		    | ![image3-2]         |

## Perspective Transformation
The perspective to parallel transform matrix is computed by *Camera.set_transformation()*. 

We pass four corners of a parallel trapezoid obtained from a straight lane on an undistorted camera image, and its corresponding rectangle in the parallel space to *cv2.getPerspectiveTransform()*. The inverse transformation is obtained by swapping the two set of corners.

To simply subsequent processing, the parallel trapezoid chosen was formed by the lane, the top and bottom of the image after cropping. 

The coordinates were:
```
(260, 0), (565, 0), (1060, 220), (720, 220).
```
When the image is transformed, the resulting rectangle will have coordinated:
```
(720, 0), (1060, 0), (1060, 220), (720, 220)
```

In the pipeline, the final image is scaled to the original image's height. This is not absolutely necessary.

## The Detection Pipeline (Single Image)

The detection pipeline is performed by *LaneDetector.detect()*. It contains the following steps:

* Undistort image: use the undistoration parameters generated in the calibration to undistort an image
* Crop the top and bottom portion of the image that are not used in the detection process in order to speed up lane detection. This step must be done after the undistoration step as the undistoration parameters are ontained with full camera images.
* Extract lane lines binary image using sobel, HSV/HLS colorspace thresholding.
* Parallel transformation the binary image
* Find left and right lane points using sliding windows
* Transfer the points back to the perspective image space
* Polyfit the points with a second order polynomial if the points are 5 or more, otherwise fit the points with polylines
* Draw the lane using fill polygon
* Overlay the lane on the undistorted image and display the image.

### Undistort Image

To demonstrate this step, the following images show the result of appling the distortion correction to one of the test images:
![image4]

And the undistorted image:
![image4-1]

### Cropping Image

I decided to crop top and bottom portion of an image that are not the interest of lane detection in order to speed up the detection process. This has to be done after the undistortion step as the undistoration parameters were computed using the original image resolution. Performing cropping before the undistortion will distort the subsequent detection.

### Extract Binary Image

I used a combination of sobelx, sobely, sobel gradient magnitude, HLS and HSV color thresholds to generate a binary image.

|           		    |                     |
|:-----------------:|:-------------------:|
| The binary image  | ![image5-1]  		    |
| Sobel X       	  | ![image5-2]         |
| Sobel Y    		    | ![image5-3]         |
| Sobel Magnitude	  | ![image5-4]         |
| HLS        		    | ![image5-5]         |
| HSV       	    	| ![image5-6]         |
| Wrap       		    | ![image5-7]         |
| Sliding Windows   | ![image5-8]         |

The HSV threshold filters image by white and yellow colors. It contains three ranges. The first range is for pure white, the second is for near white which can be any color with low saturation, and the third is for yellow:
``` 
[(0, 0, 120), (0, 0, 255)]
[(0, 0, 220), (180, 20, 255)]
[(18, 80, 120), (25, 255, 255)]
```

### Perspective Transformation

Transform images from the perspective space to the parrall space is performed by *Camera.parallel()*.
The inverse of the process is provided by *Camera.perspective()*.
The example of the transformation can be see in the second last image of the above example.

### Finding Lane

The lane finding is performed by *LaneDetector.scan()*. It use a sliding window to trace the lane. And excample is shown in the last image of the above example.

To fit the point found in the above sliding process, I call *numpy.polyfit()* to compute a second order polynomial if the number of points detected is more than 3, otherwise a linear polynomial is computed with the same function (though polylines might be more appropriate for 3 points)

### Radius of curvature of the lane

Thought not yet used, *Lane.curverature()* provide curverature compution for a given point.


### The Results

An example result image is shown:

![image6]

### Pipeline (video)

Here's a [link to my video result](./examples/project_video.mp4)

### Discussion

The detection pipeline works reasonably fine for the project_video.mp4, but does not work very well for the challenge videos. More work has to be done in fine tunning the parameters, and better detection algorithms which will require a lot more time.
