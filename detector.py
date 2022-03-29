import os
import cv2
import copy
import time
import math
import socket
import shutil
import random
import numpy as np
import pickle as pkl
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import KNeighborsClassifier

import globals, client
from encoder import Scaler, Sampling
from tracker import *

random.seed(42)
tf.compat.v1.disable_eager_execution()
tf_config = tf.compat.v1.ConfigProto()
tf_config.gpu_options.per_process_gpu_memory_fraction = 0.1 # GPU最大使用率
sess = tf.compat.v1.Session(config=tf_config)

class Detector():
    def __init__(self, dataset, fps, input_file):
        self.server_ip = '192.168.65.27'
        self.server_port = 8080
        self.dataset = dataset
        self.download_unseen_file()

        yolo = self.load_yolo(dataset)
        yolo.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        yolo.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
        custom_objects = {'Scaler': Scaler, 'Sampling': Sampling}

        self.attr_type = 'cms'
        self.track_dis = 150
        self.nms_threshold = 0.5
        self.model_ft = load_model(f'./model/{dataset}/FineTuneResNet101.h5')
        self.model = cv2.dnn_DetectionModel(yolo)
        self.model.setInputParams(size=(416, 416), scale=1/256)
        self.encoder = load_model(f'./model/{dataset}/encoder_ft_{self.attr_type}.h5', custom_objects)
        self.classes = self.read_class()
        self.classes_cnt = [0 for i in range(len(self.classes))]
        self.tracker = Tracker(self.track_dis)
        self.fps = fps
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.alpha = 0.7
        self.img_size = 224
        self.batch_size = 1
        self.set_arg()
        self.ce_mean = self.calculate_ce_mean()
        self.label_pos = 1  # reconstruct 時決定 label 在上還是下, 1代表在下
        self.input_file = input_file
        self.save_unseen_path = './tmp_unseen'
        self.build_dir(f'{self.save_unseen_path}')
        self.build_dir(f'{self.save_unseen_path}/{dataset}')
        self.clear_dir(f'{self.save_unseen_path}/{dataset}/{input_file}')
        self.reconstruct_repeat = []

    def load_yolo(self, dataset):
        yolo = cv2.dnn.readNet(f'./yolo/yolov4-{dataset}_best.weights',
                               f'./yolo/yolov4-{dataset}.cfg')
        return yolo

    def download_unseen_file(self):
        print('-' * 25)
        self.client = client.Client(self.server_ip, self.server_port, 1)
        self.unseen_file = f'./data/{self.dataset}/{self.client.host_name}_unseen_info.pkl'
        self.client.start(self.unseen_file)
        print('-' * 25)

    def upload_unseen_file(self):
        print('-' * 25)
        self.client = client.Client(self.server_ip, self.server_port, 0)
        self.client.start(self.unseen_file)

    # self.classNum 存最後的 classNum (因為 classNum 有可能會成長)
    def set_arg(self):
        if self.dataset == 'obj':
            self.original_classNum = 9
            self.classNum = self.original_classNum
            self.threshold = 0.555  # 0.56

        elif self.dataset == 'face':
            self.original_classNum = 5
            self.classNum = self.original_classNum
            self.threshold = 0.5

    def save_unseen_ce(self):
        hostName = socket.gethostname()
        unseen_ce = self.ce_mean[self.original_classNum:].tolist()
        unseen_num = self.classes_cnt[self.original_classNum:]
        unseen_name = self.classes[self.original_classNum:]

        # merge unseen_ce, unseen_num, unseen_name
        unseen_info = [[] for _ in range(len(unseen_name))]
        for i in range(len(unseen_name)):
            unseen_info[i] += unseen_ce[i]
            unseen_info[i].append(unseen_name[i])
            unseen_info[i].append(unseen_num[i])
        print(f'{self.dataset} 的 unseen_info :', np.array(unseen_info).shape)
        self.save_pkl(f'./data/{self.dataset}/{hostName}_unseen_info.pkl', unseen_info)

    def build_dir(self, path):
        if not os.path.isdir(path):
            os.mkdir(path)

    def clear_dir(self, path):
        if os.path.isdir(path):
            shutil.rmtree(path)
        os.mkdir(path)

    def read_class(self):
        with open(f'./data/{self.dataset}/trainclasses.txt', 'r') as f:
            classes = f.read().splitlines()
        return classes

    def save_pkl(self, fileName, data):
        file = open(fileName, 'wb')
        pkl.dump(data, file)
        file.close()

    def calculate_ce_mean(self):
        filePath = f'./data/{self.dataset}/seen_ce.pkl'

        # 檢查 seen ce file 是否存在
        if os.path.isfile(filePath):
            file = open(filePath, 'rb')
            return pkl.load(file)

        data_train = np.load(f'./data/{self.dataset}/feature_label_attr/train/train_feature_ft.npy')
        label_train = np.load(f'./data/{self.dataset}/feature_label_attr/train/train_label.npy')

        seen_predict = self.encoder.predict(data_train)
        seen_attr = [[] for _ in range(self.original_classNum)]
        ft_mean = [[] for _ in range(self.original_classNum)]
        count_class = [0] * self.original_classNum

        for idx in range(len(seen_predict)):
            l = label_train[idx]
            if len(seen_attr[l]):
                seen_attr[l] += np.array(seen_predict[idx])
                ft_mean[l] += np.array(data_train[idx])
            else:
                seen_attr[l] = np.array(copy.deepcopy(seen_predict[idx]))
                ft_mean[l] = np.array(copy.deepcopy(data_train[idx]))

            count_class[l] += 1

        for i in range(self.original_classNum):
            seen_attr[i] = seen_attr[i] / count_class[i]
            ft_mean[i] = ft_mean[i] / count_class[i]
        seen_attr = np.array(seen_attr) # ce_mean
        seen_ft  = np.array(ft_mean) # ft_mean

        # 沒用 p-learning 則註解掉
        # seen_merge = [[] for _ in range(len(seen_ft))]
        # for i in range(len(seen_ft)):
        #     seen_merge[i] = np.concatenate([seen_ft[i], seen_attr[i]])
        # seen_merge = np.array(seen_merge) # merge ft_mean & ce_mean
        # print(seen_merge[0][0])

        # save seen ce
        # self.save_pkl(f'./data/{self.dataset}/seen_ce.pkl', seen_merge)
        self.save_pkl(f'./data/{self.dataset}/seen_ce.pkl', seen_ft)

        # return seen_merge
        return seen_ft

    def predict_label(self, input_ft):
        neigh = KNeighborsClassifier(n_neighbors=1)
        neigh.fit(self.ce_mean, range(0, self.classNum))
        input_ft = np.expand_dims(input_ft, 0)
        predicted_label = neigh.predict(input_ft)
        label = predicted_label[0]
        write = 0

        similarity = cosine_similarity(self.ce_mean[predicted_label], input_ft)[0][0]

        # 如果不夠像, 則新增 unseen class 為新的 train data
        if similarity < self.threshold:
            write = 1
            self.classes_cnt.append(1)
            self.classNum += 1
            label = self.classNum - 1
            unseen_idx = self.classNum - self.original_classNum

            if self.dataset == 'obj':
                self.classes.append(f'Unseen_{unseen_idx}({self.dataset})')
                print(f'新增 Unseen_{unseen_idx}({self.dataset})')
            else:
                self.classes.append(f'Stranger_{unseen_idx}({self.dataset})')
                print(f'新增 Stranger_{unseen_idx}({self.dataset})')

            self.ce_mean = np.append(self.ce_mean, input_ft, axis=0)
            path = f'{self.save_unseen_path}/{self.dataset}/{self.input_file}/{self.classes[-1]}'
            self.build_dir(path)

        # 若為原先 train data 以外的 class(new train(unseen)), 則更新 ce mean
        elif label >= self.original_classNum:
            write = 1
            ce_sum = self.ce_mean[label] * self.classes_cnt[label] + input_ft[0]
            self.classes_cnt[label] += 1
            self.ce_mean[label] = ce_sum / self.classes_cnt[label]
            unseen_idx = self.classNum - self.original_classNum + 1

            if self.dataset == 'obj':
                print(f'更新 Unseen_{unseen_idx}({self.dataset}) 之 ce mean')
            else:
                print(f'更新 Stranger_{unseen_idx}({self.dataset}) 之 ce mean')

        # 若預測為 seen label, 則不更新 seen ce, 單純計算 cnt
        else:
            self.classes_cnt[label] += 1

        return label, similarity, write

    def get_class(self):
        return self.classes

    def get_class_cnt(self):
        return self.classes_cnt

    def write_img(self, classname, filename, img):
        cv2.imwrite(f'{self.save_unseen_path}/{self.dataset}/{self.input_file}/{classname}/{filename}.jpg', img)

    def detect(self, img):
        pass

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

                # P-learning
                obj_resnet_ft = self.model_ft.predict(data)
                # print(obj_resnet_ft.shape)
                # obj_attr = self.encoder.predict(obj_resnet_ft)
                # obj_merge = np.concatenate([obj_resnet_ft[0], obj_attr[0]]) # merge visual ft, semantic ft(CE)

                # predict label
                # obj_predict_label, prob, write = self.predict_label(obj_merge)
                obj_predict_label, prob, write = self.predict_label(obj_resnet_ft[0])

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

                # P-learning
                face_resnet_ft = self.model_ft.predict(data)
                face_attr = self.encoder.predict(face_resnet_ft)
                face_merge = np.concatenate([face_resnet_ft[0], face_attr[0]])

                # predict label
                face_predict_label, prob, write = self.predict_label(face_merge)

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
