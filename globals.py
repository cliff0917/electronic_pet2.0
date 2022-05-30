from detectors.obj_detector import Obj_detector
from detectors.face_detector import Face_detector

def initialize(obj_dataset, face_dataset, fps, input_file, frame_num):
    global obj_detector, face_detector, bbox_record, cnt, total_frame
    obj_detector = Obj_detector(obj_dataset, fps, input_file)
    face_detector = Face_detector(face_dataset, fps, input_file)
    bbox_record = []
    cnt = 0
    total_frame = frame_num
    