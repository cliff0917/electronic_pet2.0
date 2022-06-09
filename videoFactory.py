from videoStream import *

def VideoFactoryInterface(video_path, input_file):
    extension = input_file.split('.')[-1]
    className = {'mp4': Mp4Video, 'avi': AviVideo}
    return className[extension](video_path, input_file, extension)

class FactoryInterface(ABC):
    @abstractmethod
    def setDecodeType(self):
        pass

class Mp4VideoFactory(FactoryInterface):
    def setDecodeType(self):
        self.fourcc = cv2.VideoWriter_fourcc(*'mp4v')

class AviVideoFactory(FactoryInterface):
    def setDecodeType(self):
        self.fourcc = cv2.VideoWriter_fourcc(*'XVID')

class Mp4Video(VideoStream, Mp4VideoFactory):
    def __init__(self, video_path, input_file, extension):
        super().__init__(video_path, input_file, extension)

class AviVideo(VideoStream, AviVideoFactory):
    def __init__(self, video_path, input_file, extension):
        super().__init__(video_path, input_file, extension)