import warnings

warnings.filterwarnings("ignore", category=Warning)

import os
import cv2

from videoFactory import VideoFactoryInterface

def predict(input_file):
    video_path = './temp/my-upload'
    video = VideoFactoryInterface(video_path, input_file)
    video.run()
    os.remove(f'{video_path}/{input_file}')
    return f'{input_file} 已完成'
