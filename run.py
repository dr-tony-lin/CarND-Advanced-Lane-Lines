'''
Lane detetion
'''
import glob
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from moviepy.editor import VideoFileClip

from lane_detection import LaneDetector
from config import config
import utils

test_videos_output = "video_outputs/"
clip_name = None
clip_seq = 0

detector = LaneDetector(config)
config.test = False

def process_image(image):
    '''
    Callback from video clip
    '''
    global clip_name, clip_seq, detector
    if clip_name:
        mpimg.imsave(test_videos_output + "{0}{1}.jpg".format(clip_name, clip_seq), image)

    image, _, _, _, _, _, _, _, _, _ = detector.detect(image)
    overlay = utils.draw_lane(image.shape, config.crop[0], image.shape[0], 30, detector.left_lane, detector.right_lane)
    image = utils.weighted_img(image, overlay)

    if clip_name:
        mpimg.imsave(test_videos_output + "{0}{1}-detect.jpg".format(clip_name, clip_seq), image)
        clip_seq += 1
    return image

if not os.path.exists(test_videos_output):
    os.makedirs(test_videos_output)

for name in glob.glob("*.mp4"):
    print("Processing: {} ...".format(name))
    config.set(name)
    detector.reset()
    output = test_videos_output + name
    if config.save_video_images:
        clip_name = name
        clip_seq = 0
    clip = VideoFileClip(name)
    new_clip = clip.fl_image(process_image) #NOTE: this function expects color images!!
    new_clip.write_videofile(output, audio=False)
