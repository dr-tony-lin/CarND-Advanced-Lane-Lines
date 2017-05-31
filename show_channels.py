import glob
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from config import config

for name in glob.glob(config.test_image_folder + "*.jpg"):
    image = mpimg.imread(name)
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)

    f, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, figsize=(12, 6))
    f.tight_layout()
    ax1.imshow(hsv[:, : ,0], cmap='gray')
    ax1.set_title('HSV - H', fontsize=20)
    ax2.imshow(hsv[:, : ,1], cmap='gray')
    ax2.set_title('HSV - S', fontsize=20)
    ax3.imshow(hsv[:, : ,2], cmap='gray')
    ax3.set_title('HSV - V', fontsize=20)
    ax4.imshow(hls[:, : ,0], cmap='gray')
    ax4.set_title('HLS - H', fontsize=20)
    # ax5.imshow(hls[:, : ,1], cmap='gray')
    # ax5.set_title('HLS - L', fontsize=20)
    ax6.imshow(hls[:, : ,2], cmap='gray')
    ax6.set_title('HLS - S', fontsize=20)
    hsv[:, : ,2] = np.sqrt(hsv[:, : ,2])
    gray = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    gray = cv2.cvtColor(gray, cv2.COLOR_RGB2GRAY)
    ax5.imshow(gray, cmap='gray')
    ax5.set_title('HSV SAME Light', fontsize=20)
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    plt.show()
