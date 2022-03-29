from detector import *

def initialize(obj_dataset, face_dataset, fps, input_file): 
    global obj_detector, face_detector, bbox_record, cnt
    obj_detector = Obj_detector(obj_dataset, fps, input_file)
    face_detector = Face_detector(face_dataset, fps, input_file)
    bbox_record = []
    cnt = 0
