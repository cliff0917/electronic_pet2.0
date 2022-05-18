import cv2
import math
import numpy as np
from tensorflow.keras.applications.resnet import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import globals
from detectors.detector import Detector

class Face_detector(Detector):
    def __init__(self, dataset, fps, input_file):
        super().__init__(dataset, fps, input_file)

    def detect(self, original_img, personbox, img, color):
        class_ids, confidences, boxs = self.model.detect(img, 0.5, self.nms_threshold)

        # 如未偵測到人臉, 直接 return
        if confidences == ():
            self.tracker.incre_no_dete()
            return original_img
        else:
            indexs = np.argmax(confidences) # 一個 person 最多一個 face

        if self.tracker.get_no_dete() >= self.fps:
            self.tracker.discard_pre()

        indexs = np.array(indexs)
        overlay = original_img.copy()

        boxs_info = []
        image_gen = ImageDataGenerator(preprocessing_function=preprocess_input)
        for i in indexs.flatten():
            x, y, w, h = boxs[i]

            # 將 person 部份的臉部座標對到原本 img 的座標
            original_x = personbox[0] + x
            original_y = personbox[1] + y
            #cv2.rectangle(output, (x,y),(x+w, y+h), color, 1)

            success = False

            if self.tracker.get_first() != 1:
                self.tracker.get_curbbox(original_x+(w/2), original_y+(h/2))
                success, _, index = self.tracker.tracking_success() # face color 不須改變, 跟 obj color 一樣

            # 標出臉部
            cv2.rectangle(overlay, (original_x, original_y), (original_x + w, original_y + h), color, 2)

            if success == False:
                face = img[y:y+h,x:x+w]
                rgb_face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)

                resize_face = cv2.resize(rgb_face, (self.img_size, self.img_size))
                resize_face = np.expand_dims(resize_face, 0)

                face_gen = image_gen.flow(
                    resize_face,
                    batch_size = self.batch_size,
                    seed=42
                )
                data = face_gen.next()
                face_resnet_ft = self.model_ft.predict(data)

                # P-learning
                if self.p_learning == True:
                    face_attr = self.encoder.predict(face_resnet_ft)
                    face_merge = np.concatenate([face_resnet_ft[0], face_attr[0]])
                    face_predict_label, prob, write = self.predict_label(face_merge) # predict label
                else:
                    face_predict_label, prob, write = self.predict_label(face_resnet_ft[0]) # predict label

                name = self.classes[face_predict_label]
                prob = round(math.floor(prob * 10000) / 10000)

                if write == 1:
                    self.write_img(name, self.classes_cnt[face_predict_label], face)

                boxs_info.append([original_x+(w/2), original_y+(h/2), color, int(face_predict_label)])

            else:
                name = self.classes[index]
                self.classes_cnt[index] += 1
                boxs_info.append([original_x+(w/2), original_y+(h/2), color, index])
                prob = 1.0

                # 如果為 unseen class 寫入 img 到 save_unseen_path
                if index >= self.original_classNum:
                    face = img[y:y+h, x:x+w]
                    self.write_img(name, self.classes_cnt[index], face)

            print(f'\t{name} : {prob}')

            # label 及 class confience 顯示在下方
            background = 14 * (len(name) + len(str(prob)) + 2)
            cv2.rectangle(overlay, (original_x, original_y + h), (original_x + background, original_y + h + 18), color, -1)
            cv2.putText(overlay, f'{name} : {prob}', (original_x,original_y + h + 15), self.font, 0.7, (255,255,255), 2)
            original_img = cv2.addWeighted(overlay, self.alpha, original_img, 1 - self.alpha, 0)

            # record bbox.txt
            globals.bbox_record[globals.cnt].append([original_x, original_y, w, h, color,
                                                     prob, self.label_pos, background, name])
            #print('新增 face bbox')

        if self.tracker.get_first() == 1:
            self.tracker.set_first(0)

        # tracker 得到該 frame 所有的 bbox_info, 之後的 frame 會根據此來決定是否 tracking
        self.tracker.get_init(boxs_info)

        return original_img