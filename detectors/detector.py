import os
import cv2
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

import globals
from tracker import *
from transfer import client
from encoder import Scaler, Sampling

random.seed(42)
tf.compat.v1.disable_eager_execution()
tf_config = tf.compat.v1.ConfigProto()
tf_config.gpu_options.per_process_gpu_memory_fraction = 0.1 # GPU最大使用率
sess = tf.compat.v1.Session(config=tf_config)

class Detector():
    def __init__(self, dataset, fps, input_file, transfer=False, p_learning=False, server_ip='192.168.65.27', server_port=8080):
        self.server_ip = server_ip
        self.server_port = server_port
        self.download_unseen_file(transfer)

        self.dataset = dataset
        yolo = cv2.dnn.readNet(f'./yolo/yolov4-{dataset}_best.weights',
                               f'./yolo/yolov4-{dataset}.cfg')
        yolo.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA) # yolo 用 cuda 加速
        yolo.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

        self.p_learning = p_learning
        self.p_learning_path = f'./data/{dataset}/use_p-learning.pkl' # 取得上次是否使用 p-learning
        self.attr_type = 'cms'
        self.track_dis = 150
        self.nms_threshold = 0.5
        self.model_ft = load_model(f'./model/{dataset}/FineTuneResNet101.h5')
        self.model = cv2.dnn_DetectionModel(yolo)
        self.model.setInputParams(size=(416, 416), scale=1/256)
        self.load_encoder()
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
        self.save_pkl(self.p_learning_path, self.p_learning) # 紀錄是否使用 p-learning
        self.label_pos = 1  # reconstruct 時決定 label 在上還是下, 1代表在下
        self.input_file = input_file
        self.save_unseen_path = './tmp_unseen'
        self.build_dir(f'{self.save_unseen_path}')
        self.build_dir(f'{self.save_unseen_path}/{dataset}')
        self.clear_dir(f'{self.save_unseen_path}/{dataset}/{input_file}')
        self.reconstruct_repeat = []

    def load_encoder(self):
        if self.p_learning == True:
            custom_objects = {'Scaler': Scaler, 'Sampling': Sampling}
            self.encoder = load_model(f'./model/{self.dataset}/encoder_ft_{self.attr_type}.h5', custom_objects)

    def download_unseen_file(self, transfer):
        if transfer == False:
            return

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

    def save_pkl(self, filePath, data):
        file = open(filePath, 'wb')
        pkl.dump(data, file)
        file.close()

    def load_pkl(self, filePath):
        file = open(filePath, 'rb')
        data = pkl.load(file)
        return data

    def calculate_ce_mean(self):
        filePath = f'./data/{self.dataset}/seen_ce.pkl'

        # 用 try 是因為第一次做的時候還沒有 use_p-learning.pkl
        try:
            # 先前得到的 seen ce 維度不同, 故要刪除舊檔並重新計算
            if self.load_pkl(self.p_learning_path) != self.p_learning:
                print('維度不同, 故要刪除舊檔並重新計算')
                os.remove(filePath)
        except: pass

        # 檢查 seen ce file 是否存在
        if os.path.isfile(filePath):
            return self.load_pkl(filePath)

        data_train = np.load(f'./data/{self.dataset}/feature_label_attr/train/train_feature_ft.npy')
        label_train = np.load(f'./data/{self.dataset}/feature_label_attr/train/train_label.npy')

        # 得到 feature mean
        seen_ft = self.get_mean(label_train, data_train)

        # 有用 p-learning
        if self.p_learning == True:
            seen_predict = self.encoder.predict(data_train)
            seen_attr = self.get_mean(label_train, seen_predict)

            # merge ft_mean & ce_mean
            seen_merge = [[] for _ in range(len(seen_ft))]
            for i in range(len(seen_ft)):
                seen_merge[i] = np.concatenate([seen_ft[i], seen_attr[i]])
            seen_merge = np.array(seen_merge)

            # save seen ce
            self.save_pkl(f'./data/{self.dataset}/seen_ce.pkl', seen_merge)
            return seen_merge

        # save seen ce
        self.save_pkl(f'./data/{self.dataset}/seen_ce.pkl', seen_ft)
        return seen_ft

    def get_mean(self, label, ft):
        lst = [[] for _ in range(self.original_classNum)]
        count_class = [0] * self.original_classNum

        for idx in range(len(label)):
            l = label[idx]
            if len(lst[l]):
                lst[l] += np.array(ft[idx])
            else:
                lst[l] = np.array(ft[idx]).copy()
            count_class[l] += 1

        for i in range(self.original_classNum):
            lst[i] = lst[i] / count_class[i]
        lst = np.array(lst) # mean
        return lst

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
