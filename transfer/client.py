import socket
import pickle

class Client():
    def __init__(self, host_ip, port, request):
        # create a socket object
        self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.host_ip = host_ip
        self.port = port
        self.host_name = socket.gethostname()
        self.request = request

        # connection to hostname on the port.
        self.client_socket.connect((host_ip, port))

    def send_file(self, file_path):
        with open(file_path, "rb") as file:
            # 直到讀到檔案結尾
            while True:
                # read file data
                file_data = file.read(4194304)  # 最多一次讀 4MB

                # data 長度不為 0，表示還有 data 沒有寫入
                if file_data:
                    self.client_socket.send(file_data)
                else:
                    print(self.file_name, "傳輸成功")
                    file.close()
                    break

    def receive_file(self, file_path):
        flag = int(self.client_socket.recv(1).decode())

        if flag == 1:
            data = b''
            while True:
                packet = self.client_socket.recv(4194304) # 一次最多收 4MB
                if len(packet) == 0:
                    break
                data += packet
            data_arr = pickle.loads(data) # 將 bytes 轉為 list
            print(f'{file_path} 有被修改，從 server 上 download ce')
            self.write_pkl(file_path, data_arr)

        elif flag == 0:
            print(f'{file_path} 未被修改')
        else:
            print('server 中還未有此主機的 unseen file')

    def write_pkl(self, file_path, data):
        file = open(file_path, 'wb')
        pickle.dump(data, file)
        file.close()
        print(f'{file_path} 已寫入\n')

    def start(self, file_path):
        # get dataset
        dataset = file_path.split('/')[-2]
        print('dataset =', dataset)

        datasetLen = str(len(dataset))
        self.client_socket.send(datasetLen.encode())
        self.client_socket.send(dataset.encode())

        # get file name
        self.file_name = file_path.split('/')[-1]
        print('file_name =', self.file_name)

        fileLen = str(len(self.file_name))
        self.client_socket.send(fileLen.encode())
        self.client_socket.send(self.file_name.encode())

        # request = 0 代表 upload ce 到 server
        # request = 1 代表 download server ce, 並初始化 detector
        self.client_socket.send(str(self.request).encode())

        if self.request == 0:
            self.send_file(file_path)
        else:
            self.receive_file(file_path)
        self.client_socket.close()