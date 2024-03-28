import socket
import struct
import time
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp
import torch
from queue import *
import os
import glob
import torch
import sys

sys.path.append(r'K:\aio_radar')

class RealTimeCollector:

    def __init__(
        self, 
        ip_address='192.168.33.30', 
        port=4098, 
        output_file='data.bin',
        status=1,
        debug=False,
        is_bind=True,
        
        ):
        self.status = status                    #状态变量
        self.debug = debug                      #调试位
        self.IP_ADDRESS = ip_address            #IP地址
        self.PORT = port                        #端口号
        self.output_file = output_file          #输出文件
        self.udp_cnt = 0                        #UDP数据报计数
        self.total_payload_bytes_received = 0   #接收到的有效负载字节数
        self.udp_socket = None                  #UDP套接字  
        
        self.c = 3.0e8          # 光速
        
        self.n_TX = 1                       # TX天线通道总数
        self.n_RX = 4                       # RX天线通道总数
        self.n_samples = 64                 # 采样点数
        self.n_chirps = 255                 # 每帧脉冲数
        self.fs = 2.5e6                     # 采样频率
        self.f0 = 77e9                      # 初始频率
        self.K = 99.987e12                  # 调频斜率
        self.B = 3999.48e6                  # 调频带宽
        self.Tc = 140e-6                    # chirp总周期
        self.lambda_ = self.c / self.f0     # 雷达信号波长
        self.numLanes = 4                   # 通道数
        
        self.n_frames = 30                   # 帧数
        
        self.N = 64             # 1D FFT点数
        self.M = 64             # 2D FFT点数
        self.Q = 64             # 3D FFT点数
        
        # 初始化UDP套接字
        self.udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.udp_socket.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 40000)
        
        if is_bind:
            self.udp_socket.bind((self.IP_ADDRESS, self.PORT))
        
        # 设置绘图中文显示
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用中文黑体字体
        plt.rcParams['axes.unicode_minus'] = False  # 解决坐标轴负号显示问题
    
        # 绘图参数        
        self.distance_axis = np.arange(0, self.N) * self.fs * self.c / (2 * self.K * self.N)        # 距离轴
        self.velocity_axis = np.arange(-self.M/2, self.M/2) * self.lambda_ / (self.Tc * self.M / 2) # 速度轴
        self.angle_axis = np.arange(-self.Q/2, self.Q/2)                                            # 角度轴
        
        # 其他参数(暂时不用)
        self.single_frame_size = self.n_TX * self.n_RX * self.n_samples * self.n_chirps * 2
        self.single_frame_length = self.n_TX * self.n_RX * self.n_samples * self.n_chirps * 2  # 单帧数据长度
    
    def udp_listener(self, udp_queue=None):
        """
        侦听UDP数据报并将其存储到UDP数据报队列中的函数。
        参数:
        - udp_queue(Queue): 用于存储UDP数据报的队列。
            - dtype: bytes
        返回:
        None
        """
        print(f"正在监听来自 {self.IP_ADDRESS}:{self.PORT} 的UDP数据...")
        while self.get_status():
            try:
                data, _ = self.udp_socket.recvfrom(1477)
                udp_queue.put(data)
                
                # 统计
                self._udp_statistics(data) 
                
            except KeyboardInterrupt:
                self.set_status(0)
                self.udp_socket.close()
                break
        
        print(f"接收到的数据报数量: {self.udp_cnt}")
        print(f"接收到的有效负载字节数: {self.total_payload_bytes_received}")
    
    def udp_storage(self,udp_queue=None, filename=None):
        """
        将UDP数据报存储到文件的函数。

        参数:
        - udp_queue(Queue): 用于存储UDP数据报的队列。
        - filename(str): 用于存储UDP数据报的文件名。
        """
        
        if filename is None:
            filename = "data.bin"
            self.clear_file(filename)
        
        print(self.status)
        print("正在存储UDP数据报...")
        
        while self.status:
            try:
                if udp_queue.empty():
                    continue
                else:
                    seq_num, payload_size, payload = self._udp_parser(udp_queue.get())
                    self.process_udp_storage(payload, filename)
                    # data = udp_queue.get()
                    # self.process_udp_storage(data, filename)
            except KeyboardInterrupt:
                self.set_status(0)
                break
    
    def udp_2frame(self, udp_queue=None, frame_queue=None):
        """
        将 UDP 数据转换为帧数据的函数。

        参数:
        - udp_data_queue(Queue): 用于存储 UDP 数据报的队列。
        - frame_queue(Queue): 用于存储帧数据的队列。

        返回:
        Queue: 存储帧数据的队列

        """
        frame = np.array([],dtype=np.int16)
        
        while self.status:
            try:
                if udp_queue.qsize() == 0:
                    continue
                else:
                    seq_num, payload_size, payload = self._udp_parser(udp_queue.get())
                    
                    payload = np.frombuffer(payload, dtype=np.int16)
                    if len(frame) + len(payload) >= self.single_frame_length:
                        
                        temp_len = self.single_frame_length - len(frame)
                        frame = np.hstack((frame, payload[:temp_len]))
                        
                        frame = np.squeeze(self.reshape_frame(frame))
                        
                        frame_queue.put(frame)
                        frame = np.array([],dtype=np.int16)
                        frame = np.hstack((frame, payload[temp_len:]))
                        
                    else:
                        frame = np.hstack((frame, payload))
            except KeyboardInterrupt:
                self.set_status(0)
                break
        
        return frame_queue
    
    def frame_handler(self, frame_queue=None, is_draw=False):
        """
        帧数据处理函数。

        参数:
        - frame_queue: 存储帧数据的队列。如果未提供，将使用类属性中的默认队列。
        - is_draw: 是否绘制数据。默认为 False。

        Returns:
            None
        
        """
        while self.get_status:
            try:
                if frame_queue.empty():
                    continue
                else:
                    range_profile, speed_profile, angle_profile = self.process_frame(frame_queue.get(),clutter_removal='avg')
                    if is_draw:
                        self.frame_display(range_profile, speed_profile, angle_profile)
            except KeyboardInterrupt:
                self.set_status(0)
                break
    
    def frame_display(self, range_profile=None,speed_profile=None,angle_profile=None):
        # [ ] frame_display设计未完成,需要根据实际情况进行修改
        plt.ion()  # 将画图模式改为交互模式
        plt.clf()  # 清空画布上的所有内容
        
        plt.plot(self.distance_axis, np.abs(range_profile[1,:,1]))
        
        plt.pause(0.005)    
    
    def process_udp_storage(self,data=None, filename=None):
        """
        处理UDP数据报的存储。

        参数:
        - data (bytes): 需要存储的UDP数据报。
        - filename (str, optional): 要保存的文件名。
        返回:
        None
        """
        with open(filename, 'ab') as f:
            f.write(data)
    
    def process_frame(self, frame, clutter_removal=None, is_squeeze=False):
        """
        对雷达帧数据进行处理。

        Parameters:
        - frame (numpy.ndarray): 输入的雷达帧数据。
            - shape: (n_RX, n_samples, n_chirps)
            - or shape: (n_RX, n_samples, n_chirps, n_frames)
        - clutter_removal (str, optional): 静态杂波滤除的选项。可选值为 'avg'（平均滤波）或 'mti'（移动目标指示滤波）。默认为 None，不执行额外的滤波处理。

        Returns:
        - range_profile (numpy.ndarray): 距离特征。
        - speed_profile (numpy.ndarray): 速度特征。
        - angle_profile (numpy.ndarray): 角度特征。
        """
        # 如果输入的帧数据是三维的，则将其转换为四维
        if frame.ndim == 3:
            frame = frame[:, :, :, np.newaxis]
        
        range_profile = np.fft.fft(frame, self.N, axis=1)
        if clutter_removal == 'avg':
            range_profile = range_profile - np.mean(range_profile, axis=2)[:, :, np.newaxis, :]
            
        elif clutter_removal == 'mti':
            range_profile = range_profile - np.roll(range_profile, 1, axis=2)
        else:
            pass
        
        speed_profile = np.fft.fftshift(np.fft.fft(range_profile, self.M, axis=2), axes=2)
        
        angle_profile = np.fft.fftshift(np.fft.fft(speed_profile, self.Q, axis=0), axes=0)
        
        if is_squeeze:
            range_profile = np.squeeze(np.mean(np.abs(range_profile), axis=(0, 2))).T
            speed_profile = np.squeeze(np.mean(np.abs(speed_profile), axis=(0, 1))).T
            angle_profile = np.squeeze(np.mean(np.abs(angle_profile), axis=(1, 2))).T
        
        return range_profile, speed_profile, angle_profile

    def process_file(self, filename=None, clutter_removal=None):
        """
        从文件中读取雷达帧数据并对其进行处理。

        Parameters:
        - filename (str): 文件名。
        - clutter_removal (str, optional): 静态杂波滤除的选项。可选值为 'avg'（平均滤波）或 'mti'（移动目标指示滤波）。默认为 None，不执行额外的滤波处理。

        Returns:
        - range_profile (numpy.ndarray): 距离特征。
        - speed_profile (numpy.ndarray): 速度特征。
        - angle_profile (numpy.ndarray): 角度特征。
        """
        with open(filename, 'rb') as f:
            data = f.read()
        
        frame = np.frombuffer(data, dtype=np.int16)
        frame = np.reshape(frame, (self.numLanes*2, -1), order='F')
        frame = frame[[0, 1, 2, 3], :] + 1j * frame[[4, 5, 6, 7], :]
        frame = np.reshape(frame, (self.n_RX, self.n_samples, self.n_chirps, -1), order='F')
        
        return self.process_frame(frame, clutter_removal)
    
    # 读取udp数据并到指定的大小后进行ABCnet处理
    def udp_2ABCnet(self, udp_queue=None):
        """
        将 UDP 数据转换为帧数据的函数。

        参数:
        - udp_data_queue(Queue): 用于存储 UDP 数据报的队列。
        - frame_queue(Queue): 用于存储帧数据的队列。

        返回:
        Queue: 存储帧数据的队列

        """
        
        from ABCnet import RadarGestureNet
        from ABCnet import one_hot_to_label
        
        # 加载模型
        model_path = r'K:\aio_radar\lightning_logs\version_70\checkpoints\epoch=74-step=750.ckpt'
        model_path = r'K:\aio_radar\lightning_logs\version_83\checkpoints\epoch=74-step=1050.ckpt'
        model = RadarGestureNet.load_from_checkpoint(model_path).to("cpu")
        
        print("模型加载成功！")
        
        frame_data = []
        frame_data_size = 0
        t = time.time()
        while self.status:
            if udp_queue.qsize() == 0:
                continue
            else:
                seq_num, payload_size, payload = self._udp_parser(udp_queue.get())
                payload = np.frombuffer(payload, dtype=np.int16)
                if frame_data_size + payload.size < self.single_frame_size * 30:
                    frame_data_size += payload.size
                    frame_data.append(payload)
                else:
                    # 取出刚好要填满的数据
                    temp_len = self.single_frame_size*30 - frame_data_size
                    temp_data = payload[:temp_len]
                    
                    payload = payload[temp_len:] 
                    
                    frame_data.append(temp_data)
                    
                    # 将数据连接起来
                    frame = np.concatenate(frame_data)
                    print("frame shape:", frame.shape)
                    
                    frame_data = []  # 清空列表
                    frame_data_size = 0  # 清空数据大小
                    
                    frame_data.append(payload)  # 将剩下的数据暂存在列表中
                    frame_data_size += payload.size

                    # 重组数据
                    frame = frame.reshape(self.numLanes*2, -1,order='F')
                    frame = frame[[0,1,2,3],:] + 1j*frame[[4,5,6,7],:]

                    data_radar = np.reshape(frame, (self.n_RX, self.n_samples, self.n_chirps, self.n_frames), order='F')

                    print(data_radar.shape)  # (4, 64, 255, 30)

                    # 特征提取
                    range_profile, speed_profile, angle_profile = self.process_frame(data_radar,clutter_removal='avg')

                    # 压缩维度
                    range_profile = np.squeeze(np.mean(np.abs(range_profile), axis=(0, 2))).T
                    speed_profile = np.squeeze(np.mean(np.abs(speed_profile), axis=(0, 1))).T
                    angle_profile = np.squeeze(np.mean(np.abs(angle_profile), axis=(1, 2))).T
                    
                    range_profile = torch.tensor(range_profile, dtype=torch.float32)
                    speed_profile = torch.tensor(speed_profile, dtype=torch.float32)
                    angle_profile = torch.tensor(angle_profile, dtype=torch.float32)
                    
                    
                    # 在第0维增加一个维度(如果是二维的)
                    if len(range_profile.shape) == 2:
                        range_profile = range_profile.unsqueeze(0)
                    if len(speed_profile.shape) == 2:
                        speed_profile = speed_profile.unsqueeze(0)
                    if len(angle_profile.shape) == 2:
                        angle_profile = angle_profile.unsqueeze(0)
                    
                    # out = model(range_profile, speed_profile, angle_profile)
                    # confidence = out[0][one_hot_to_label(out)].item()
                    # if confidence > 0.95:
                    #     print(out)
                    #     print("Prediction:", one_hot_to_label(out), "!!!!!!!!!!!!!!!!!!")
                    # print("confidence:",confidence)
                    # print("remaining udp_queue size:",udp_queue.qsize())
                    
                    print("time:",time.time()-t)
                    t = time.time()
    
    def get_status(self):
        """获取状态变量"""
        return self.status

    def get_udp_queue_size(self,udp_queue=None):
        """获取UDP数据报队列的大小"""
        return udp_queue.qsize()
    
    def get_frame_queue_size(frame_queue=None):
        """获取帧数据队列的大小"""
        return frame_queue.qsize()

    def pop_udp_packet(udp_queue=None):
        '''从UDP数据报队列中取出一个UDP数据报'''
        return udp_queue.get()

    def pop_frame(frame_queue=None):
        '''从帧数据队列中取出一个帧数据'''
        return frame_queue.get()
    
    def generate_distance_map(self,range_profile):
        """
        生成基于给定距离幅度数据的距离图。

        参数：
        - range_profile（numpy.ndarray）：距离幅度数据。

        返回：
        - matplotlib.figure.Figure：包含距离图的生成图形。

        """
        fig = plt.figure()
        distance_map = fig.add_subplot(111)
        distance_map.plot(self.distance_axis, np.abs(range_profile))
        # 设置标题
        distance_map.set_title('距离图')
        # 设置坐标轴标签
        distance_map.set_xlabel('距离')
        distance_map.set_ylabel('幅度')
        
        return fig

    def generate_distance_speed_map(self, speed_profile=None):
        """
        生成距离-速度图。

        Parameters:
        - speed_profile (numpy.ndarray): 速度剖面数据，二维数组，其中行表示距离，列表示速度。

        Returns:
        - matplotlib.figure.Figure: 生成的距离-速度图的Matplotlib Figure 对象。
        
        注意：此函数使用了Matplotlib库来生成图表。确保在调用此函数之前导入Matplotlib。
        """
        fig = plt.figure()
        plt.imshow(np.abs(speed_profile.T), cmap='jet', aspect='auto', 
                extent=[self.distance_axis[0], self.distance_axis[-1], self.velocity_axis[0], self.velocity_axis[-1]])
        plt.title('距离-速度图')
        plt.xlabel('距离')
        plt.ylabel('速度')
        return fig
    
    def generate_udp_packet(self, filename=None, max_file_num=None):
        """
        从文件中生成UDP数据报。
        
        参数:
        - filename (str): 用于生成UDP数据报的文件名或目录名。
        - max_file_num (int): 生成UDP数据报的最大文件数量。默认为 None。
        返回:
        data (list): UDP数据报列表。
            -dtype: bytes
        """
        
        if os.path.isdir(filename):
            data = b''
            bin_files = sorted(glob.glob(os.path.join(filename, '*.bin')))
            if max_file_num is None:
                max_file_num = len(bin_files)
            for i, bin_file in enumerate(bin_files):
                print("当前i:",i,"当前bin_file:",bin_file)
                with open(bin_file, "rb") as file:
                    data += file.read()
                if (i+1) == max_file_num:
                    break
        else:
            with open(filename, "rb") as file:
                data = file.read()
        
        # 分割数据为长度为 1456 字节的数据包
        data = [data[i:i+1456] for i in range(0, len(data), 1456)]
        
        # 为每个数据包添加消息序列号和捕获大小
        for i in range(len(data)):
            data[i] = struct.pack('>I6s', i, b'000000') + data[i]
        
        return data

    def dca1000evm_controller(self):
        """
        用于控制DCA1000EVM
        """
        # [ ] dca1000evm_controller
        pass

    def dca1000evm_simulator(self, filename=None,max_file_num=10, ip_address=None, port=None, frame_rate=10, is_repeat=False):
        """
        模拟DCA1000EVM，生成UDP数据报。
        
        参数:
        - filename (str): 用来生成UDP数据报的文件名。
        - max_file_num (int): 生成UDP数据报的最大文件数量。默认为 10。
        - ip_address (str): IP地址。
        - port (int): 端口号。
        - frame_rate (int): 帧率。默认为 10 帧/秒。
        - is_repeat (bool): 是否重复发送。默认为 False。
        返回:
        None
        """
        print("filename:",filename)
        data = self.generate_udp_packet(filename,max_file_num)
        print("len(data):",len(data))
        while self.get_status():
            try:
                for i in range(len(data)):
                    index = i%len(data)
                    # print("index:",index)
                    
                    self.udp_socket.sendto(data[i%len(data)], (ip_address, port))
                    # time.sleep(1/frame_rate/len(data))
                    # if i%5 == 0:
                    #     time.sleep(0.001)
                # time.sleep(1/frame_rate*10)
                if not is_repeat:
                    # 等待10秒
                    time.sleep(10)
                    break
            
            except KeyboardInterrupt:
                time.sleep(10)
                self.set_status(0)
                break
    
    def reshape_frame(self, frame=None):
        """
        将帧数据转换为适合处理的格式。
        
        参数:
        - frame (bytes): 帧数据。
        
        返回:
        - frame (numpy.ndarray): 转换后的帧数据。
            - shape: (n_RX, n_samples, n_chirps, n_frames)
        """
        # 实部虚部结合
        frame = np.reshape(frame, (self.numLanes*2, -1), order='F')
        frame = frame[[0, 1, 2, 3], :] + 1j * frame[[4, 5, 6, 7], :]
        # 重组成帧格式
        frame = np.reshape(frame, (self.n_RX, self.n_samples, self.n_chirps,-1), order='F')
        return frame
    
    def create_udp_queue(self,is_shared=False):
        """
        创建UDP数据报队列
        
        参数:
        - is_shared (bool): 是否为共享队列。默认为 False。
        返回:
        Queue: UDP数据报队列
        """
        if is_shared:
            return mp.Manager().Queue()
        else:
            return Queue()
    
    def create_frame_queue(self,is_shared=False):
        """
        创建帧数据队列
        
        参数:
        - is_shared (bool): 是否为共享队列。默认为 False。
        返回:
        Queue: 帧数据队列
        """
        if is_shared:
            return mp.Manager().Queue()
        else:
            return Queue()
    
    def clear_file(self,filename=None):
        """清空文件"""
        with open(filename,'wb') as f:
            f.truncate()
    
    def clear_udp_data_queue(self,udp_queue=None):
        '''清空UDP数据报队列'''
        udp_queue.queue.clear()

    def clear_frame_queue(self,frame_queue=None):
        '''清空帧数据队列'''
        frame_queue.queue.clear()
    
    def set_radar_params(self, n_TX=1, n_RX=4, n_samples=64, n_chirps=255, fs=2.5e6, f0=77e9, K=99.987e12, B=3999.48e6, Tc=140e-6, lambda_=None):
        """
        设置雷达参数。
        Parameters:
            n_TX (int): TX天线通道的总数。
            n_RX (int): RX天线通道的总数。
            n_samples (int): 采样点数。
            n_chirps (int): 每帧的脉冲数。
            fs (float): 采样频率。
            f0 (float): 初始频率。
            K (float): 调频斜率。
            B (float): 带宽。
            Tc (float): 脉冲宽度。
            lambda_ (float): 波长。
        Returns:
            None
        """
        self.n_TX = n_TX
        self.n_RX = n_RX
        self.n_samples = n_samples
        self.n_chirps = n_chirps
        self.fs = fs
        self.f0 = f0
        self.K = K
        self.B = B
        self.Tc = Tc
        self.lambda_ = lambda_
        
        self._update_plot_params()
    
    def set_fft_params(self, N=None, M=None, Q=None):
        """
        设置FFT相关参数，包括距离轴、速度轴和角度轴。

        Parameters:
        - N (int, optional): FFT的点数（距离轴的分辨率）。如果未指定，则保持当前值不变。
        - M (int, optional): FFT的点数（速度轴的分辨率）。如果未指定，则保持当前值不变。
        - Q (int, optional): FFT的点数（角度轴的分辨率）。如果未指定，则保持当前值不变。

        Returns:
        None
        注意：此函数会更新距离轴、速度轴和角度轴的参数，以确保它们与新的FFT参数一致。
        """
        if self.N != N:
            self.N = N
        if self.M != M:
            self.M = M
        if self.Q != Q:
            self.Q = Q
        
        self._update_plot_params()

    def set_status(self, status=None):
        """设置状态变量"""
        self.status=status
    
    def _update_plot_params(self):
        """更新绘图参数"""
        self.distance_axis = np.arange(0, self.N) * self.fs * self.c / (2 * self.K * self.N)
        self.velocity_axis = np.arange(-self.M/2, self.M/2) * self.lambda_ / (self.Tc * self.M / 2)
        self.angle_axis = np.arange(-self.Q/2, self.Q/2)

    def _udp_parser(self, udp_packet):
        """
        解析UDP报文，提取其中的不同部分信息。

        Parameters:
        - udp_packet (bytes): 待解析的UDP报文。

        Returns:
        - sequence_number_hex (str): 消息序列号的十六进制表示。
        - capture_size_hex (str): 捕获大小的十六进制表示。
        - payload (bytes): 报文的有效负载部分。
        """
        sequence_number = udp_packet[:4]          # 消息序列号
        capture_size = udp_packet[4:10]          # 捕获大小
        payload = udp_packet[10:]                # 有效负载

        sequence_number_hex = ' '.join(format(byte, '02X') for byte in sequence_number)
        capture_size_hex = ' '.join(format(byte, '02X') for byte in capture_size)

        return sequence_number_hex, capture_size_hex, payload
    
    def _udp_statistics(self, data):
        '''统计接收到的UDP数据报数量和有效负载字节数'''
        self.udp_cnt=self.udp_cnt+1 
        self.total_payload_bytes_received += len(data) - 10
        # 创建帧处理函数

    def test(self,frame_queue=None, is_draw=False, model_path=None):
        """
        帧数据处理函数。

        参数:
        - frame_queue: 存储帧数据的队列。如果未提供，将使用类属性中的默认队列。
        - is_draw: 是否绘制数据。默认为 False。
        - model_path: 模型路径。默认为 None。

        Returns:
            None
        
        """
        from AAnet import RadarGestureNet
        from AAnet import one_hot_to_label

        # 加载模型
        if model_path is None:
            # model_path = r'I:\aio\aio_radar\aio_network\lightning_logs\version_216\checkpoints\epoch=74-step=525.ckpt'
            model_path = r'I:\aio\aio_radar\aio_network\lightning_logs\version_281\checkpoints\epoch=74-step=300.ckpt'
            model_path = r'K:\aio_radar\lightning_logs\version_5\checkpoints\epoch=74-step=525.ckpt'
            model_path = r'K:\aio_radar\lightning_logs\version_43\checkpoints\epoch=149-step=150.ckpt'
            model_path = r'K:\aio_radar\lightning_logs\version_44\checkpoints\epoch=149-step=150.ckpt'
        model = RadarGestureNet.load_from_checkpoint(model_path).to("cpu")
        
        while self.get_status:
            try:
                if frame_queue.empty():
                    continue
                else:
                        
                    # 每30帧进行一次预测
                    if frame_queue.qsize() < 30:
                        continue
                    else:
                        frame_list = np.zeros((4,64,255,30),dtype=complex)
                        
                        for i in range(0,30):
                            frame_list[:,:,:,i] = frame_queue.get()
                        
                        range_profile, speed_profile, angle_profile = self.process_frame(frame_list,clutter_removal='avg')
                        
                        # feature = range_profile
                        feature = angle_profile
                        
                        feature = np.squeeze(np.mean(np.abs(feature), axis=(1,2))).T
                        
                        print(feature.shape)
                        
                        # 预测
                        tensor_feature = torch.tensor(feature, dtype=torch.float32)
                        # 在第0维增加一个维度(如果是二维的)
                        if len(tensor_feature.shape) == 2:
                            tensor_feature = tensor_feature.unsqueeze(0)
                            
                        print(feature.shape)
                        # plt.ion()  # 将画图模式改为交互模式
                        # plt.clf()  # 清空画布上的所有内容
                        # plt.plot(np.sum(feature, axis=0))
                        

                        print(model.encoder(tensor_feature))
                        print(one_hot_to_label(model.encoder(tensor_feature)))
            except KeyboardInterrupt:
                break
    
    def ABCnet_process(self,frame_queue=None,udp_queue=None, is_draw=False, model_path=None):
        """
        帧数据处理函数。

        参数:
        - frame_queue: 存储帧数据的队列。如果未提供，将使用类属性中的默认队列。
        - udp_queue: 存储UDP数据报的队列。如果未提供，将使用类属性中的默认队列。
        - is_draw: 是否绘制数据。默认为 False。
        - model_path: 模型路径。默认为 None。

        Returns:
            None
        
        """
        from ABCnet import RadarGestureNet
        from ABCnet import one_hot_to_label

        # 加载模型
        if model_path is None:
            # model_path = r'K:\aio_radar\lightning_logs\version_45\checkpoints\epoch=74-step=75.ckpt'
            model_path = r'K:\aio_radar\lightning_logs\version_47\checkpoints\epoch=74-step=75.ckpt'
            model_path = r'K:\aio_radar\lightning_logs\version_83\checkpoints\epoch=74-step=1050.ckpt'
        model = RadarGestureNet.load_from_checkpoint(model_path).to("cpu")
        
        frame_list = np.zeros((4,64,255,30),dtype=complex)
        
        while self.get_status:
            try:
                if frame_queue.empty():
                    continue
                else:
                    # 每30帧进行一次预测
                    if frame_queue.qsize() < 30:
                        continue
                    else:
                        for i in range(0,30):
                            frame_list[:,:,:,i] = frame_queue.get()
                        
                        
                        temp_range_profile, temp_speed_profile, temp_angle_profile = self.process_frame(frame_list,clutter_removal='avg')
        
                        # 压缩维度
                        range_profile = torch.tensor(np.squeeze(np.mean(np.abs(temp_range_profile), axis=(0, 2))).T, dtype=torch.float32)
                        speed_profile = torch.tensor(np.squeeze(np.mean(np.abs(temp_speed_profile), axis=(0, 1))).T, dtype=torch.float32)
                        angle_profile = torch.tensor(np.squeeze(np.mean(np.abs(temp_angle_profile), axis=(1, 2))).T, dtype=torch.float32)
                        
                        
                        # 预测
                        # 在第0维增加一个维度(如果是二维的)
                        if len(range_profile.shape) == 2:
                            range_profile = range_profile.unsqueeze(0)
                        if len(speed_profile.shape) == 2:
                            speed_profile = speed_profile.unsqueeze(0)
                        if len(angle_profile.shape) == 2:
                            angle_profile = angle_profile.unsqueeze(0)
                        
                        out = model(range_profile, speed_profile, angle_profile)
                        confidence = out[0][one_hot_to_label(out)].item()
                        if confidence > 0.55:
                            print(out)
                            print("Prediction:", one_hot_to_label(out), "!!!!!!!!!!!!!!!!!!")
                        print("remaining udp_queue size:",udp_queue.qsize())
                        print("remaining frame_queue size:",frame_queue.qsize())
                        print("confidence:",confidence)
                        
            except KeyboardInterrupt:
                break

