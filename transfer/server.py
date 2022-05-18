import os
import socket
import pickle
import datetime
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class Server():
    def __init__(self, port):
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.equivalence_class = [] # 紀錄不同 partner 間 unseen class 的等價關係(只紀錄 unseen class)
        self.equivalence_hostName = [] # 紀錄上述對應的 host(partner) name

        # get local machine name
        self.host_name = socket.gethostname()
        self.host_ip = socket.gethostbyname(self.host_name)
        self.port = port
        self.threshold = 0.7
        self.modified = {}

        # bind to the port
        self.server_socket.bind((self.host_name, port))

        # queue up to 5 requests
        self.server_socket.listen(5)

        print('Host Name :', self.host_name)
        print('Host IP :', self.host_ip)

    def start(self):
        while True:
            self.client_socket, client_ip = self.server_socket.accept()
            print('-' * 25)
            print(f'Got a connection from {client_ip}')

            connectTime = datetime.datetime.now()
            print('Connected time:', connectTime)

            # dataset名稱都為個位數
            datasetLen = self.client_socket.recv(1)
            dataset = self.client_socket.recv(int(datasetLen)).decode()
            #print(dataset)

            # unseen檔名長度都為兩位數
            fileLen = self.client_socket.recv(2)
            file_name = self.client_socket.recv(int(fileLen)).decode()
            #print(file_name)

            # 如果還沒有該 dataset 存放 unseen info 的 dir，則建一個給它
            self.path = './' + dataset
            if not os.path.isdir(self.path):
                os.mkdir(self.path)

            # 要寫入 or 檢查是否被修改的 file
            file_path = self.path + '/' + file_name

            # 檢查 client 是要初始化 detector 還是 upload ce
            request = int(self.client_socket.recv(1).decode())

            # 初始化 detector
            if request == 1:
                flag = self.check_modified(file_path)
                if flag == 1:
                    self.client_socket.send(b'1')
                    print(f'{file_path} 有被修改，server 將修改後的 ce 傳輸給該主機')
                    self.send_file(file_path)
                elif flag == 0:
                    self.client_socket.send(b'0')
                    print(f'{file_path} 未被修改')
                else:
                    self.client_socket.send(b'2')
                    print(f'server 中還未有 {file_path}')

            # client 上傳 ce 到 server
            else:
                self.receive_file(file_path)
            self.client_socket.close()
        self.server_socket.close()

    def check_modified(self, file_path):
        try:
            return self.modified[file_path]

        # modified 中沒有 key=file_path, catch error
        except:
            return 2

    def receive_file(self, file_path):
        data = b''
        while True:
            packet = self.client_socket.recv(4194304)  # 一次最多收 4MB
            if len(packet) == 0:
                break
            data += packet
        data_arr = pickle.loads(data) # 將 bytes 轉為 list
        #print(data_arr)

        # write file
        self.write_pkl(file_path, data_arr)
        self.modified[file_path] = 0 # file 設為未修改

        # class aggregate
        fileName = file_path.split('/')[-1]
        self.class_aggregate(fileName)

    def send_file(self, file_path):
        try:
            with open(file_path, "rb") as file:
                # 直到讀到檔案結尾
                while True:
                    # read file data
                    file_data = file.read(4194304)  # 最多一次讀 4MB

                    # data 長度不為 0，表示還有 data 沒有寫入
                    if file_data:
                        self.client_socket.send(file_data)
                    else:
                        print(file_path, "傳輸成功")
                        self.modified[file_path] = 0 # 將 file 設為 unmodified
                        file.close()
                        break
        except Exception as e:
            print("傳輸異常:", e)

    def addToEquivClass(self, host1, className1, host2, className2):
        findFirst, firstIdx = self.isInEquivClass(host1, className1)
        findSecond, secondIdx = self.isInEquivClass(host2, className2)

        # 兩個 partner 之 unseen class 都不在 EC 中
        if findFirst == 0 and findSecond == 0:
            self.equivalence_class.append([className1, className2])
            self.equivalence_hostName.append([host1, host2])
            print(f'[{className1}, {className2}] 加入 EC 中')

        # p1 之 unseen class 在 EC 中, 但 p2 的不在
        elif findFirst == 1 and findSecond == 0:
            self.equivalence_class[firstIdx].append(className2)
            self.equivalence_hostName[firstIdx].append(host2)
            print(f'{className2} 加入 {className1} 的 EC 中')

        # p2 之 unseen class 在 EC 中, 但 p1 的不在
        elif findFirst == 0 and findSecond == 1:
            self.equivalence_class[secondIdx].append(className1)
            self.equivalence_hostName[secondIdx].append(host1)
            print(f'{className1} 加入 {className1} 的 EC 中')

        # 兩個 partner 之 unseen class 都在 EC 中
        else:
            self.equivalence_class[firstIdx] += self.equivalence_class[secondIdx]
            self.equivalence_class.pop(secondIdx)
            self.equivalence_hostName[firstIdx] += self.equivalence_hostName[secondIdx]
            self.equivalence_hostName.pop(secondIdx)
            print(f'{className1} 的 EC 合併 {className2} 的 EC')

    def isInEquivClass(self, host, className):
        ec = self.equivalence_class
        ef = self.equivalence_hostName
        for i in range(len(ec)):
            for j in range(len(ec[i])):
                if className == ec[i][j] and host == ef[i][j]:
                    return 1, i
        return 0, 0

    def class_aggregate(self, newFile):
        # 取得 path 下所有檔案與子目錄名稱
        print('New file:', newFile)
        files = os.listdir(self.path)
        files.remove(newFile)
        print('Other files:', files)

        if len(files) != 0:
            file_path1 = self.path + '/' + newFile
            host1, data1 = self.read_pkl(newFile)
            for i in range(len(files)):
                file_path2 = self.path + '/' + files[i]
                host2, data2 = self.read_pkl(files[i]) # data 為該檔案所有 unseen ce
                for m in range(len(data1)):
                    ce1, className1, classNum1 = self.split_pkl(data1[m])
                    for n in range(len(data2)):
                        ce2, className2, classNum2 = self.split_pkl(data2[n])
                        similarity = cosine_similarity(ce1, ce2)[0][0]
                        print(f'[{host1} 的 {className1}, {host2} 的 {className2}] 之相似度: {similarity}')
                        if similarity >= self.threshold:
                            print(f'*** {host1} 的 {className1} 等價 {host2} 的 {className2} ***')
                            self.addToEquivClass(host1, className1, host2, className2)
                self.write_pkl(file_path2, data2)
            self.write_pkl(file_path1, data1)
        else:
            print('目前只有一個檔案，不用做 class aggregation')
        print('*' * 25)
        print(self.equivalence_class)
        print(self.equivalence_hostName)
        print('*' * 25)

    # 檢查 ce1, ce2 是否完全一樣
    def same_ce(self, ce1, classNum1, ce2, classNum2):
        if classNum1 != classNum2:
            return False
        for i in range(len(ce1)):
            if ce1[i] != ce2[i]:
                return False
        return True

    def write_pkl(self, file_path, data):
        file = open(file_path, 'wb')
        pickle.dump(data, file)
        file.close()
        print(f'{file_path} 已寫入\n')

    # pkl 中 ce, className, classNum 放在一起， 要用下面的 split_pkl 將它們分開
    def read_pkl(self, file_name):
        host = file_name.split('_')[0]
        file_path = self.path + '/' + file_name
        with open(file_path, 'rb') as file:
            data = pickle.load(file)
        if type(data) == bytes:
            data = pickle.loads(data)
        file.close()
        return host, data

    def split_pkl(self, data):
        ce = np.expand_dims(data[:-2], 0)
        className = data[-2]
        classNum = data[-1]
        return ce, className, classNum

if __name__ == '__main__':
    port = 8080
    server = Server(port)
    server.start()