import glob
import math
import os
import platform
import threading
from threading import Thread
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.backend_bases import NavigationToolbar2
from config import config
import utils

class Cursor(object):
    def __init__(self, ax, image):
        self.ax = ax
        self.image = image

        # text location in axes coords
        self.txt = ax.text(0.7, 0.9, '', transform=ax.transAxes, color=(1, 0, 0))

    def mouse_move(self, event):
        if not event.inaxes:
            return
        y, x = int(event.xdata), int(event.ydata)
        print(x, ', ', y)
        self.txt.set_text('({0}, {1}, {2})'.format(self.image[x, y, 0], self.image[x, y, 1], self.image[x, y, 2]))
        plt.draw()

class Plot(Thread):
    def __init__(self, folder, filter):
        super().__init__()
        self.folder = folder
        self.filter = filter

    def run(self):
        ks = 5
        for name in glob.glob(self.folder + self.filter):
            image = mpimg.imread(name)
            hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)

            fig, ((ax1, ax2, ax3, ax4, ax5), (ax6, ax7, ax8, ax9, ax10)) = plt.subplots(2, 5, figsize=(24, 8))
            fig.tight_layout()
            ax1.imshow(image)
            ax1.set_title('Original', fontsize=20)
            ax2.imshow(hsv[:, :, 0], cmap='gray')
            ax2.set_title('HSV - H', fontsize=20)
            ax3.imshow(hsv[:, :, 1], cmap='gray')
            ax3.set_title('HSV - S', fontsize=20)
            ax4.imshow(hsv[:, :, 2], cmap='gray')
            ax4.set_title('HSV - V', fontsize=20)
            ax5.imshow(hls[:, :, 0], cmap='gray')
            ax5.set_title('HLS - H', fontsize=20)
            # ax5.imshow(hls[:, : ,1], cmap='gray')
            # ax5.set_title('HLS - L', fontsize=20)
            ax6.imshow(hls[:, :, 2], cmap='gray')
            ax6.set_title('HLS - S', fontsize=20)
            hsv[:, :, 2] = np.sqrt(hsv[:, :, 2])
            hsv[:, :, 2] = hsv[:, :, 2]/np.amax(hsv[:, :, 2])*255.0
            norm = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
            ax7.imshow(norm)
            ax7.set_title('HSV norm val', fontsize=20)
            hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            mask = cv2.inRange(hsv, (0, 0, 0), (180, 25, 255)) |\
                   cv2.inRange(hsv, (110, 10, 0), (128, 120, 255))
            kernel = np.ones((7, 7), np.uint8)
            # mask = cv2.erode(mask,kernel,iterations = 1)
            ax8.imshow(mask, cmap="gray")
            ax8.set_title('Road mask', fontsize=20)
            sobelx = np.absolute(cv2.Sobel(mask, cv2.CV_64F, 1, 0, ksize=5))[config.crop[0]:config.crop[1], :]
            sobely = np.absolute(cv2.Sobel(mask, cv2.CV_64F, 0, 1, ksize=5))[config.crop[0]:config.crop[1], :]
            sobelm = np.sqrt(sobelx * sobelx + sobely * sobely)
            sobelx = sobelx/(np.amax(sobelx)+1e-6)
            sobely = sobely/(np.amax(sobely)+1e-6)
            sobelm = sobelm/(np.amax(sobelm)+1e-6)
            kernel = np.ones((3, 3), np.uint8)
            sobelm = cv2.erode(sobelm, kernel, iterations=1)
            sobelm = cv2.dilate(sobelm, kernel, iterations=1)
            lane_img = utils.find_houghlines(np.uint8(sobelm/np.amax(sobelm)*255), threshold=100,
                                             min_line_len=100, max_line_gap=20, thickness=4,
                                             angle=0.15)
            ax9.imshow(lane_img, cmap='gray')
            ax9.set_title('Road', fontsize=20)
            ax10.imshow(sobelm, cmap='gray')
            ax10.set_title('Road solem', fontsize=20)
            plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
            cursor1 = Cursor(ax1, hsv)
            plt.connect('motion_notify_event', cursor1.mouse_move)
            cursor2 = Cursor(ax7, cv2.cvtColor(norm, cv2.COLOR_RGB2HSV))
            plt.connect('motion_notify_event', cursor2.mouse_move)
            plt.show()

def command_handler(command):
    '''
    Handler user command
    '''
    if command == 'exit' or command == 'x':
        print("Exiting ...")
        os._exit(0)
    else:
        print("Unknown command: {}!".format(command))

def accept_inputs(callback=command_handler):
    '''
    Accept user inputs
    callback -  the callback to call when received an input
    '''
    def _input():
        run = True
        while run:
            line = input()
            if line:
                callback(line.strip().lower())
    if platform.system() == 'Windows': # WIndows can run in a different thread
        thread = threading.Thread(target=_input)
        thread.setDaemon(True)
        thread.start()
        return thread
    else: # other environments need to run in main thread
        assert threading.current_thread() == threading.main_thread(), "accept_inputs() should be called from main thread!"
        _input()
plot = Plot(config.test_image_folder, "*.jpg")
input_thread = accept_inputs(command_handler)
plot.start()
