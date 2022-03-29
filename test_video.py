import warnings

warnings.filterwarnings("ignore", category=Warning)

import cv2
import time

from videoFactory import *
from videoStream import *

if __name__ == '__main__':
    
    #########################
    video_path = './video_test'
    input_file = '3.mp4'
    #input_file = '3.avi'
    #########################

    video = VideoFactoryInterface(video_path, input_file)
    video.run()
