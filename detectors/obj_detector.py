import cv2
import math
import numpy as np
from tensorflow.keras.applications.resnet import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import globals
from detectors.detector import Detector

class Obj_detector(Detector):
    def __init__(self, dataset, fps, input_file):
        super().__init__(dataset, fps, input_file)

    def detect(self, img):
        class_ids, confidences, boxs = self.model.detect(img, 0.2, self.nms_threshold)
        #print('boxs', boxs)

        if boxs == ():
            self.tracker.incre_no_dete()
            return img

        if self.tracker.get_no_dete() >= self.fps:
            self.tracker.discard_pre()

        # 隨機產生 n 種顏色, n 為 bbox 個數
        colors = np.random.uniform(0, 255, size = (len(boxs), 3))
        overlay = img.copy()
        output = img.copy()

        boxs_info = []
        image_gen = ImageDataGenerator(preprocessing_function=preprocess_input)

        for i in range(len(boxs)):
            x, y, w, h = boxs[i]

            success = False

            if self.tracker.get_first() != 1: # 第一個 frame 一定不用 tracking, 因此不會進到這個 if 裡, 直接做 p-learning
                self.tracker.get_curbbox(x+(w/2), y+(h/2))
                success, color, index = self.tracker.tracking_success()

            if success == False: # tracking failed, then do p-learning & knn to predict label
                obj = img[y:y+h, x:x+w]
                rgb_obj = cv2.cvtColor(obj, cv2.COLOR_BGR2RGB)

                resize_obj = cv2.resize(rgb_obj, (self.img_size, self.img_size))
                resize_obj = np.expand_dims(resize_obj, 0)

                obj_gen = image_gen.flow(
                    resize_obj,
                    batch_size = self.batch_size,
                )
                data = obj_gen.next()
                obj_resnet_ft = self.model_ft.predict(data)

                # P-learning
                if self.p_learning == True:
                    obj_attr = self.encoder.predict(obj_resnet_ft)
                    obj_merge = np.concatenate([obj_resnet_ft[0], obj_attr[0]]) # merge visual ft, semantic ft(CE)
                    obj_predict_label, prob, write = self.predict_label(obj_merge) # predict label
                else:
                    obj_predict_label, prob, write = self.predict_label(obj_resnet_ft[0]) # predict label

                label = self.classes[obj_predict_label]
                color = colors[i]
                boxs_info.append([x+(w/2), y+(h/2), color, int(obj_predict_label)])

                # 取到小數點後兩位, 直接用 round 會失敗
                prob = round(math.floor(prob * 10000) / 10000)

                if write == 1:
                    self.write_img(label, self.classes_cnt[obj_predict_label], obj)

            else: # tracking success
                label = self.classes[index] # get the label of previous frame
                self.classes_cnt[index] += 1
                boxs_info.append([x+(w/2), y+(h/2), color, index])
                prob = 1.0

                if index >= self.original_classNum: # if it is unseen class, then write img, do p-learning & update ce_mean
                    obj = img[y:y+h, x:x+w]
                    self.write_img(label, self.classes_cnt[index], obj)

            print(f'{label} : {prob}')

            cv2.rectangle(overlay, (x,y), (x+w, y+h), color, 3)
            background = 14 * (len(label) + len(str(prob)) + 2)

            # label 顯示在下方
            if y-20 < 0 or y-5 < 0:
                cv2.rectangle(overlay, (x,y), (x+background,y+17), color, -1)
                cv2.putText(overlay, f'{label} : {prob}', (x,y+12), self.font, 0.7, (255,255,255), 2)
                self.label_pos = 1

            # label 顯示在上方
            else:
                cv2.rectangle(overlay, (x,y-20), (x+background,y), color, -1)
                cv2.putText(overlay, f'{label} : {prob}', (x,y-5), self.font, 0.7, (255,255,255), 2)
                self.label_pos = 0

            output = cv2.addWeighted(overlay, self.alpha, output, 1 - self.alpha, 0)

            # record bbox.txt
            globals.bbox_record[globals.cnt].append([x, y, w, h, color, prob, self.label_pos, background, label])
            #print('新增 obj bbox')

            if label == 'person':
                person = img[y:y+h, x:x+w]
                output = globals.face_detector.detect(output, boxs[i], person, color)
                overlay = output.copy() # 讓 img 不斷疊加
        #print('box info : ', boxs_info)

        # 如果為第一張 frame 或 obj-tracking 失敗後的第一張 frame, 做完要把 first 設成 1, 之後才能進 obj-tracking
        if self.tracker.get_first() == 1:
            self.tracker.set_first(0)

        # tracker 得到該 frame 所有的 bbox_info, 並之後的 frame 會根據此來決定是否 tracking
        self.tracker.get_init(boxs_info)

        return output