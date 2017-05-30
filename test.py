'''
Lane detetion
'''
import glob
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from lane_detection import LaneDetector
from config import config
import utils

test_undistort = False
show_undistort = False
show_lane = False

if not os.path.exists(config.calibration_folder + "undistorted"):
    os.makedirs(config.calibration_folder + "undistorted")

if not os.path.exists(config.test_image_folder + "lanes"):
    os.makedirs(config.test_image_folder + "lanes")

if not os.path.exists(config.test_image_folder + "extract"):
    os.makedirs(config.test_image_folder + "extract")

if not os.path.exists(config.test_image_folder + "thresh"):
    os.makedirs(config.test_image_folder + "thresh")

detector = LaneDetector(config)

if test_undistort:
    for name in glob.glob(config.calibration_folder + "*.jpg"):
        print("Undistort : ", name)
        image = mpimg.imread(name)
        undistorted = detector.camera.undistort(image)
        mpimg.imsave(config.calibration_folder + "undistorted/" + os.path.basename(name), undistorted)
        if show_undistort:
            f, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
            f.tight_layout()
            ax1.imshow(image)
            ax1.set_title('Original Image', fontsize=20)
            ax2.imshow(undistorted)
            ax2.set_title('Undistorted', fontsize=20)
            plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
            plt.show()

for name in glob.glob(config.test_image_folder + "*.jpg"):
    print("Test : ", name)
    image = mpimg.imread(name)
    print(image.shape)
    image, visual, trasnsformed, undistort, extracted, sobelx, sobely, sobelm, hls, hsv = detector.detect(image)
    overlay = utils.draw_lane((720, 1280, 3), 470, 720, 30, detector.left_lane, detector.right_lane)
    lane_image = utils.weighted_img(image, overlay)
    mpimg.imsave(config.test_image_folder+"lanes/{}-undistort.jpg".format(os.path.basename(name)[0:-4]), image)
    mpimg.imsave(config.test_image_folder+"lanes/{}-final.jpg".format(os.path.basename(name)[0:-4]), lane_image)
    image = image[config.crop[0]:config.crop[1], :, :]
    perspective = detector.camera.parallel(image)
    if perspective is not None:
        mpimg.imsave(config.test_image_folder+"lanes/{}-persp.jpg".format(os.path.basename(name)[0:-4]), perspective)
    if visual is not None:
        mpimg.imsave(config.test_image_folder+"lanes/{}-trace.jpg".format(os.path.basename(name)[0:-4]), visual)
    if trasnsformed is not None:
        mpimg.imsave(config.test_image_folder+"lanes/{}".format(os.path.basename(name)), trasnsformed, cmap='gray')
    if extracted is not None:
        mpimg.imsave(config.test_image_folder+"extract/{}".format(os.path.basename(name)), extracted, cmap='gray')
    if sobelx is not None:
        mpimg.imsave(config.test_image_folder+"thresh/{}-sobelx.jpg".format(os.path.basename(name)[0:-4]), sobelx, cmap='gray')
    if sobely is not None:
        mpimg.imsave(config.test_image_folder+"thresh/{}-sobely.jpg".format(os.path.basename(name)[0:-4]), sobely, cmap='gray')
    if sobelm is not None:
        mpimg.imsave(config.test_image_folder+"thresh/{}-sobelm.jpg".format(os.path.basename(name)[0:-4]), sobelm, cmap='gray')
    if hls is not None:
        mpimg.imsave(config.test_image_folder+"thresh/{}-hls.jpg".format(os.path.basename(name)[0:-4]), hls, cmap='gray')
    if hsv is not None:
        mpimg.imsave(config.test_image_folder+"thresh/{}-hsv.jpg".format(os.path.basename(name)[0:-4]), hsv, cmap='gray')

    if show_lane:
        f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 8))
        f.tight_layout()

        ax1.imshow(image)
        ax1.set_title('Original Image', fontsize=20)
        ax2.imshow(threshd, cmap='gray')
        ax2.set_title('Threshold', fontsize=20)
        ax3.imshow(extracted, cmap='gray')
        ax3.set_title('Extracted', fontsize=20)
        ax4.imshow(perspective)#camera.transform(image))
        ax4.set_title('Transformed', fontsize=20)
        plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
        plt.show()
