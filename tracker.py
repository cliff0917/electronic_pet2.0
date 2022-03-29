import numpy as np

class Tracker():

    def __init__(self, dis_thre):
        self.pre_frame_record = np.array([])
        self.threshold = dis_thre**2
        self.cur_bbox = np.array([])
        self.first = 1
        self.no_dete_cnt = 0

    def tracking_success(self):
        distance = []
        for i in range(len(self.pre_frame_record)):
            tmp = self.cur_bbox - self.pre_frame_record[i][:2]
            dis = np.dot(tmp, tmp)
            distance.append(dis)
        min_dis = min(distance)
        min_label = distance.index(min_dis)

        # return 是否 tracking 成功, color, label
        if min_dis <= self.threshold:
            print('tracking成功')
            self.cur_bbox = []
            self.no_dete_cnt = 0
            return True, self.pre_frame_record[min_label][-2], int(self.pre_frame_record[min_label][-1])
        else:
            print(f'tracking失敗, dis = {min_dis}')
            self.no_dete_cnt = 0
            self.cur_bbox = []
            return False, 0, 0

    def discard_pre(self):
        self.pre_frame_record = np.array([])
        self.set_first(1)
        self.no_dete_cnt = 0
        print('捨棄 pre_info')

    def set_first(self, value):
        self.first = value

    def get_first(self):
        return self.first

    def get_init(self, boxs):
        self.pre_frame_record = np.array(boxs)
        #print('prev : ',self.pre_frame_record)

    def get_curbbox(self, x, y):
        self.cur_bbox = np.append(self.cur_bbox, [x,y])
        #print('cur : ', self.cur_bbox)

    def incre_no_dete(self):
        self.no_dete_cnt += 1

    def get_no_dete(self):
        return self.no_dete_cnt

    def get_record(self):
        return self.pre_frame_record
