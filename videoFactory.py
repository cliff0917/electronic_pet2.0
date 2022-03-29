from videoStream import *
  
def VideoFactoryInterface(video_path, input_file):
    extension = input_file.split('.')[-1]
    className = {'mp4': Mp4Video, 'avi': AviVideo}
    return className[extension](video_path, input_file, extension)