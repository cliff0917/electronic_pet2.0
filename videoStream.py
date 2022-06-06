import cv2
import copy
import time
import numpy as np
from abc import ABC, abstractmethod
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics.pairwise import cosine_similarity
from rq import get_current_job

import globals

class VideoStream(ABC):
    def __init__(self, video_path, input_file, extension):
        self.video_path = video_path
        self.input_file = input_file
        self.file_name = input_file.split('.')[0]
        self.tmp_file = f'{self.file_name}_tmp.{extension}' # the file before reconstruct
        self.output_file = f'{self.file_name}_final.{extension}'    # the file after reconstruct
        self.cap = cv2.VideoCapture(f'{video_path}/{input_file}')
        self.length = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.frame_num = self.cap.get(cv2.CAP_PROP_FRAME_COUNT)
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.alpha = 0.7
        self.setVideoType()
        self.setOutputType()

    def setOutputType(self):
        self.tmp_out = cv2.VideoWriter(
            f'{self.video_path}/{self.tmp_file}',
            self.fourcc, self.fps, (self.length, self.width))

        self.reconstruct_out = cv2.VideoWriter(
            f'{self.video_path}/{self.output_file}',
            self.fourcc, self.fps, (self.length, self.width))

    def run(self):
        globals.initialize('obj', 'face', self.fps, self.input_file, self.frame_num)
        globals.bbox_record = [[] for _ in range(int(self.frame_num))]

        job = get_current_job()
        job.meta["progress"] = 0
        job.save_meta()

        avg_process_time = 0.0
        start = time.time()
        while self.cap.isOpened():
            ret, img = self.cap.read()
            if not ret:
                break
            print('-' * 25)
            print(f'frame : {globals.cnt}')
            self.tmp_out.write(globals.obj_detector.detect(img))
            globals.cnt += 1
            job.meta["progress"] = 100 * globals.cnt / self.frame_num
            job.save_meta()
        end = time.time()

        self.cap.release()
        self.tmp_out.release()
        total_time = round(end - start, 2)
        self.record_file(total_time, 'w+')

        print('-' * 25)
        print(f'花費 : {total_time}秒')
        print('FPS :', round((self.frame_num / total_time), 2))
        print('-' * 25)

        self.double_check() # 校正 unseen class
        self.video_reconstruct()    # 校正完做 reconstruct

        # reconstruct 完上傳 unseen file, 用 try 是避免 server 沒開導致 error
        try:
            globals.obj_detector.upload_unseen_file()
            globals.face_detector.upload_unseen_file()
        except: pass

    def double_check(self):
        print('obj:')
        globals.obj_detector.reconstruct_repeat = self.check(globals.obj_detector)
        print('-' * 25)
        print('face:')
        globals.face_detector.reconstruct_repeat = self.check(globals.face_detector)
        print('-' * 25)

    def check(self, detector):
        merge = {} # 存要 rename 的 unseen class 和其對應的 seen class
        f = open(f'{self.video_path}/{self.input_file}_record.txt', 'a')
        f.write('\n')

        if detector.threshold >= 0.6:
            threshold = detector.threshold
        else:
            threshold = detector.threshold + 0.2

        neigh = KNeighborsClassifier(n_neighbors=1)
        neigh.fit(detector.ce_mean[:detector.original_classNum], range(0, detector.original_classNum))

        # check 每個 unseen 是否為 seen
        detector.reconstruct_repeat = [0 for i in range(detector.classNum)]
        for i in range(detector.original_classNum, detector.classNum):
            unseen_ce = np.expand_dims(detector.ce_mean[i], 0) # unseen ce mean
            label = int(neigh.predict(unseen_ce))
            print(f'離{detector.classes[i]}最近的為 : ', detector.classes[label])
            closest_ce = np.array(detector.ce_mean[label]).reshape(1, -1) # closest seen ce mean
            similarity = cosine_similarity(closest_ce, unseen_ce)[0][0]
            print('相似度 : ', similarity)

            if similarity > threshold:
                unseen_name = detector.classes[i]
                replace_name = detector.classes[label]
                detector.classes[i] = detector.classes[label]
                print(f'*** {unseen_name} 校正成為 {replace_name} ***')
                f.write(f'\n*** {unseen_name} 校正成為 {replace_name} ***\n')
                merge[unseen_name] = replace_name
                detector.classes_cnt[label] += detector.classes_cnt[i]
                detector.reconstruct_repeat[i] = 1
        f.close()

        print(f'classNum: {detector.classNum}')
        print(detector.classes)

        # merge 不為空
        if merge:
            # 修正錯誤的 label
            for frame_info in globals.bbox_record:
                for bbox_info in frame_info:
                    for unseen_name, replace_name in merge.items():
                        if bbox_info[-1] == unseen_name:
                            bbox_info[-1] = replace_name # 修正 label
                            prob = bbox_info[-3]
                            bbox_info[-2] = 14 * (len(replace_name) + len(str(prob)) + 2) # 修正 background 長度
        return detector.reconstruct_repeat

    def video_reconstruct(self):
        self.cap = cv2.VideoCapture(f'{self.video_path}/{self.input_file}') # 重新打開 input file
        job = get_current_job()
        job.meta["reconstruct"] = 0
        job.save_meta()

        i = 0
        start = time.time()
        while self.cap.isOpened():
            ret, img = self.cap.read()
            if not ret:
                break

            overlay = img.copy()
            output = img.copy()
            for bbox_info in globals.bbox_record[i]:
                x, y, w, h, color, prob, label_pos, background, label = bbox_info
                prob = round(prob, 2)
                cv2.rectangle(overlay, (x, y), (x + w, y + h), color, 3)

                # label 顯示在上方
                if label_pos == 0:
                    cv2.rectangle(overlay, (x,y-20), (x+background,y), color, -1)
                    cv2.putText(overlay,
                                f'{label} : {prob}', (x, y - 5),
                                self.font, 0.7, (255, 255, 255), 2)
                else:
                    cv2.rectangle(overlay, (x, y), (x+background,y+17), color, -1)
                    cv2.putText(overlay,
                                f'{label} : {prob}', (x, y + 12),
                                self.font, 0.7, (255, 255, 255), 2)
                # 影像疊加
                output = cv2.addWeighted(overlay, self.alpha, output, 1 - self.alpha, 0)
                overlay = output.copy()

            self.reconstruct_out.write(output)
            i += 1
            job.meta["reconstruct"] = 100 * i / self.frame_num
            job.save_meta()

        end = time.time()
        self.cap.release()
        self.reconstruct_out.release()
        total_time = round(end - start, 2)
        print(f'reconstruct 花費 : {total_time}秒')
        self.record_file(total_time, 'a')

        # reconstruct 完儲存 unseen ce
        globals.obj_detector.save_unseen_ce()
        globals.face_detector.save_unseen_ce()

    def record_file(self, total_time, mode):
        obj_classes = globals.obj_detector.classes
        face_classes = globals.face_detector.classes
        obj_classes_cnt = globals.obj_detector.classes_cnt
        face_classes_cnt = globals.face_detector.classes_cnt

        # 排序 cnt 大到小的 index
        obj_sort_index = np.argsort(np.array(obj_classes_cnt), kind='stable')[::-1]
        face_sort_index = np.argsort(np.array(face_classes_cnt), kind='stable')[::-1]

        file = open(f'{self.video_path}/{self.input_file}_record.txt', mode)

        if mode == 'a':
            file.write('-' * 25 + '\n\n')
            file.write('after reconstruct:\n\n')
        else:
            file.write('-' * 25 + '\n')
            file.write('before reconstruct:\n\n')

        for i in range(len(obj_sort_index)):
            obj_sort_label = obj_sort_index[i]
            cnt = obj_classes_cnt[obj_sort_label]

            try:
                if cnt == 0 or globals.obj_detector.reconstruct_repeat[obj_sort_label] == 1:
                    continue
            except:
                pass

            obj_name = obj_classes[obj_sort_label]

            if obj_name == 'person':
                file.write(f'{i + 1}. {obj_name}: {cnt}次\n')

                for j in range(len(face_sort_index)):
                        face_sort_label = face_sort_index[j]
                        face_cnt = face_classes_cnt[face_sort_label]

                        try:
                            if face_cnt == 0 or globals.face_detector.reconstruct_repeat[face_sort_label] == 1:
                                continue
                        except:
                            pass

                        face_name = face_classes[face_sort_label]
                        file.write(f'\t({j + 1}) {face_name}: {face_cnt}次\n')
            else:
                file.write(f'{i + 1}. {obj_name}: { obj_classes_cnt[obj_sort_label]}次\n')
        file.write(f'\n總共花費 {total_time} 秒')
        file.close()
