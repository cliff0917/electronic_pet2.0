import warnings

warnings.filterwarnings("ignore", category=Warning)

import cv2

from videoFactory import VideoFactoryInterface

def predict(input_file):
    video_path = './temp/my-upload'
    video = VideoFactoryInterface(video_path, input_file)
    video.run()
    return f'{input_file} 已完成, 進行 reconstruct'
