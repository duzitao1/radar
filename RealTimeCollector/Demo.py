import multiprocessing as mp
from RealTimeCollector import RealTimeCollector

class Demo:
    def __init__(self):
        pass
    
    # 模拟DCA1000EVM并使用AAnet预测手势
    def demo_01(self):
        
        # 初始化队列
        udp_queue = mp.Manager().Queue()
        frame_queue = mp.Manager().Queue()
        status = mp.Manager().Value('b', True)
        
        # 创建实时数据采集器
        collector = RealTimeCollector(ip_address='127.0.0.1', port=4098, output_file='data.bin',status=status)
        
        # 创建DC1000EVM模拟器进程
        dca1000evm_simulator_process = mp.Process(target=collector.dca1000evm_simulator, args=(
            # "K:/手势识别数据集/2/2_3_Raw_0.bin",
            # "K:/手势识别数据集/2/2_5_Raw_0.bin",
            # "I:/aio/aio_radar/aio_gusture/dataset/2023_9_11/水杯快速向前向后",
            # "I:/aio/aio_radar/aio_gusture/dataset/2023_9_17/4",
            "K:/dataset/2024_3_7/2",
            # "K:/手势识别数据集/2/",
            10,
            '127.0.0.1', 
            4098,
            10,
            True))
        # 创建UDP监听进程
        udp_listener_process = mp.Process(target=collector.udp_listener, args=(udp_queue,))
        # 创建UDP转帧进程
        udp_2frame_process = mp.Process(target=collector.udp_2frame, args=(udp_queue,frame_queue))
        # 创建帧处理进程
        frame_handler_process = mp.Process(target=collector.test, args=(frame_queue,True))
        
        
        
        dca1000evm_simulator_process.start()       # 启动DC1000EVM模拟器
        udp_listener_process.start()               # 启动UDP监听器
        udp_2frame_process.start()                 # 启动UDP转帧器
        frame_handler_process.start()              # 启动帧处理器

        # 等待进程结束
        try:
            dca1000evm_simulator_process.join()
        except KeyboardInterrupt:
            dca1000evm_simulator_process.terminate()
            udp_listener_process.terminate()
            udp_2frame_process.terminate()
            frame_handler_process.terminate()
        
        print("剩余未处理udp数据包数量：",udp_queue.qsize())
        print("剩余未处理帧数据数量：",frame_queue.qsize())
        print("demo_01 end")

    # 实时采集数据并使用AAnet预测手势
    def demo_02(self):
        # 初始化队列
        udp_queue = mp.Manager().Queue()
        frame_queue = mp.Manager().Queue()
        status = mp.Manager().Value('b', True)
        
        # 创建实时数据采集器
        collector = RealTimeCollector(ip_address='192.168.33.30', port=4098, output_file='data.bin',status=status)
        
        # 创建UDP监听进程
        udp_listener_process = mp.Process(target=collector.udp_listener, args=(udp_queue,))
        # 创建UDP转帧进程
        udp_2frame_process = mp.Process(target=collector.udp_2frame, args=(udp_queue,frame_queue))
        # 创建帧处理进程
        frame_handler_process = mp.Process(target=collector.test, args=(frame_queue,True))
        
        
        
        udp_listener_process.start()               # 启动UDP监听器
        udp_2frame_process.start()                 # 启动UDP转帧器
        frame_handler_process.start()              # 启动帧处理器

        # 等待进程结束
        try:
            udp_listener_process.join()
        except KeyboardInterrupt:
            udp_2frame_process.terminate()
            frame_handler_process.terminate()
        
        print("剩余未处理udp数据包数量：",udp_queue.qsize())
        print("剩余未处理帧数据数量：",frame_queue.qsize())
        print("demo_02 end")
    
    # 模拟生成数据并存储
    def demo_03(self):
        # 初始化队列
        udp_queue = mp.Manager().Queue()
        frame_queue = mp.Manager().Queue()
        status = mp.Manager().Value('b', True)
        
        # 创建实时数据采集器
        collector = RealTimeCollector(ip_address='127.0.0.1', port=4098, output_file='data.bin',status=status)
        
        # 创建DC1000EVM模拟器进程
        dca1000evm_simulator_process = mp.Process(target=collector.dca1000evm_simulator, args=(
            # "K:/手势识别数据集/2/2_3_Raw_0.bin",
            # "K:/手势识别数据集/2/2_5_Raw_0.bin",
            # "I:/aio/aio_radar/aio_gusture/dataset/2023_9_11/水杯快速向前向后",
            # "I:/aio/aio_radar/aio_gusture/dataset/2023_9_17/4",
            # "K:/dataset/2024_3_7/1",
            # "K:/手势识别数据集/2/",
            10,
            '127.0.0.1', 
            4098,
            10,
            True
            ))
        # 创建UDP监听进程
        udp_listener_process = mp.Process(target=collector.udp_listener, args=(udp_queue,))
        # 创建数据存储进程
        udp_storage_process = mp.Process(target=collector.udp_storage, args=(udp_queue,))
        
        
        
        dca1000evm_simulator_process.start()       # 启动DC1000EVM模拟器
        udp_listener_process.start()               # 启动UDP监听器
        udp_storage_process.start()                # 启动数据存储器

        # 等待进程结束
        try:
            dca1000evm_simulator_process.join()
        except KeyboardInterrupt:
            dca1000evm_simulator_process.terminate()
            udp_listener_process.terminate()
            udp_storage_process.terminate()
        
        print("剩余未处理udp数据包数量：",udp_queue.qsize())
        print("剩余未处理帧数据数量：",frame_queue.qsize())
        print("demo_03 end")
    
    # 实时采集数据并存储
    def demo_04(self):
        # 初始化队列
        udp_queue = mp.Manager().Queue()
        frame_queue = mp.Manager().Queue()
        status = mp.Manager().Value('b', True)
        
        # 创建实时数据采集器
        collector = RealTimeCollector(ip_address='127.0.0.1', port=4098, output_file='data.bin',status=status)
        
        # 创建UDP监听进程
        udp_listener_process = mp.Process(target=collector.udp_listener, args=(udp_queue,))
        # 创建数据存储进程
        udp_storage_process = mp.Process(target=collector.udp_storage, args=(udp_queue,))
        
        
        
        udp_listener_process.start()               # 启动UDP监听器
        udp_storage_process.start()                # 启动数据存储器

        # 等待进程结束
        try:
            udp_listener_process.join()
        except KeyboardInterrupt:
            udp_storage_process.terminate()
        
        print("剩余未处理udp数据包数量：",udp_queue.qsize())
        print("剩余未处理帧数据数量：",frame_queue.qsize())
        print("demo_04 end")
    
    # 模拟DCA1000EVM并使用ABCnet预测手势
    def demo_05(self):
        
        # 初始化队列
        udp_queue = mp.Manager().Queue()
        frame_queue = mp.Manager().Queue()
        status = mp.Manager().Value('b', True)
        
        # 创建实时数据采集器
        collector = RealTimeCollector(ip_address='127.0.0.1', port=4098, output_file='data.bin',status=status)
        
        # 创建DC1000EVM模拟器进程
        dca1000evm_simulator_process = mp.Process(target=collector.dca1000evm_simulator, args=(
            # "K:/手势识别数据集/2/2_3_Raw_0.bin",
            # "K:/手势识别数据集/2/2_5_Raw_0.bin",
            # "I:/aio/aio_radar/aio_gusture/dataset/2023_9_11/水杯快速向前向后",
            # "I:/aio/aio_radar/aio_gusture/dataset/2023_9_17/4",
            "K:/dataset/2024_3_7/2",
            # "K:/手势识别数据集/2/",
            10,
            '127.0.0.1', 
            4098,
            10,
            True))
        # 创建UDP监听进程
        udp_listener_process = mp.Process(target=collector.udp_listener, args=(udp_queue,))
        # 创建UDP转帧进程
        udp_2frame_process = mp.Process(target=collector.udp_2frame, args=(udp_queue,frame_queue))
        # 创建帧处理进程
        frame_handler_process = mp.Process(target=collector.ABCnet_process, args=(frame_queue,True))
        
        
        
        dca1000evm_simulator_process.start()       # 启动DC1000EVM模拟器
        udp_listener_process.start()               # 启动UDP监听器
        udp_2frame_process.start()                 # 启动UDP转帧器
        frame_handler_process.start()              # 启动帧处理器

        # 等待进程结束
        try:
            dca1000evm_simulator_process.join()
        except KeyboardInterrupt:
            dca1000evm_simulator_process.terminate()
            udp_listener_process.terminate()
            udp_2frame_process.terminate()
            frame_handler_process.terminate()
        
        print("剩余未处理udp数据包数量：",udp_queue.qsize())
        print("剩余未处理帧数据数量：",frame_queue.qsize())
        print("demo_05 end")
    
    # 实时采集数据并使用ABCnet预测手势
    def demo_06(self):
        
        # 初始化队列
        udp_queue = mp.Manager().Queue()
        frame_queue = mp.Manager().Queue()
        status = mp.Manager().Value('b', True)
        
        # 创建实时数据采集器
        collector = RealTimeCollector(ip_address='192.168.33.30', port=4098, output_file='data.bin',status=status)
        
        # 创建UDP监听进程
        udp_listener_process = mp.Process(target=collector.udp_listener, args=(udp_queue,))
        # 创建UDP转帧进程
        udp_2frame_process = mp.Process(target=collector.udp_2frame, args=(udp_queue,frame_queue))
        # 创建帧处理进程
        frame_handler_process = mp.Process(target=collector.ABCnet_process, args=(frame_queue,udp_queue,True))
        
        udp_listener_process.start()               # 启动UDP监听器
        udp_2frame_process.start()                 # 启动UDP转帧器
        frame_handler_process.start()              # 启动帧处理器

        # 等待进程结束
        try:
            udp_listener_process.join()
        except KeyboardInterrupt:
            udp_2frame_process.terminate()
            frame_handler_process.terminate()
        
        print("剩余未处理udp数据包数量：",udp_queue.qsize())
        print("剩余未处理帧数据数量：",frame_queue.qsize())
        print("demo_06 end")
    
    # 模拟生成数据并直接从udp_queue中取数据用ABCnet预测手势
    def demo_07(self):
        
        # 初始化队列
        udp_queue = mp.Manager().Queue()
        frame_queue = mp.Manager().Queue()
        status = mp.Manager().Value('b', True)
        
        # 创建实时数据采集器
        rtc = RealTimeCollector(ip_address='127.0.0.1', port=4098, output_file='data.bin',status=status)
        
        # 创建DC1000EVM模拟器进程
        dca1000evm_simulator_process = mp.Process(target=rtc.dca1000evm_simulator, args=(
            # "K:/手势识别数据集/2/2_3_Raw_0.bin",
            # "K:/手势识别数据集/2/2_5_Raw_0.bin",
            # "I:/aio/aio_radar/aio_gusture/dataset/2023_9_11/水杯快速向前向后",
            # "I:/aio/aio_radar/aio_gusture/dataset/2023_9_17/4",
            "K:/dataset/2024_3_7/2",
            # "K:/手势识别数据集/2/",
            10,
            '127.0.0.1', 
            4098,
            20,
            True))
        # 创建UDP监听进程
        udp_listener_process = mp.Process(target=rtc.udp_listener, args=(udp_queue,))
        # 创建帧处理进程
        frame_handler_process = mp.Process(target=rtc.udp_2ABCnet, args=(udp_queue,))
        
        
        
        dca1000evm_simulator_process.start()       # 启动DC1000EVM模拟器
        udp_listener_process.start()               # 启动UDP监听器
        frame_handler_process.start()              # 启动帧处理器

        # 等待进程结束
        try:
            dca1000evm_simulator_process.join()
        except KeyboardInterrupt:
            dca1000evm_simulator_process.terminate()
            udp_listener_process.terminate()
            frame_handler_process.terminate()
        
        print("剩余未处理udp数据包数量：",udp_queue.qsize())
        print("剩余未处理帧数据数量：",frame_queue.qsize())
        print("demo_07 end")
    
    # 实时采集数据并直接从udp_queue中取数据用ABCnet预测手势
    def demo_08(self):
        
        # 初始化队列
        udp_queue = mp.Manager().Queue()
        frame_queue = mp.Manager().Queue()
        status = mp.Manager().Value('b', True)
        
        # 创建实时数据采集器
        rtc = RealTimeCollector(ip_address='192.168.33.30', port=4098, output_file='data.bin',status=status)
        
        # 创建UDP监听进程
        udp_listener_process = mp.Process(target=rtc.udp_listener, args=(udp_queue,))
        # 创建帧处理进程
        frame_handler_process = mp.Process(target=rtc.udp_2ABCnet, args=(udp_queue,))
        
        udp_listener_process.start()               # 启动UDP监听器
        frame_handler_process.start()              # 启动帧处理器

        # 等待进程结束
        try:
            udp_listener_process.join()
        except KeyboardInterrupt:
            frame_handler_process.terminate()
        
        print("剩余未处理udp数据包数量：",udp_queue.qsize())
        print("剩余未处理帧数据数量：",frame_queue.qsize())
        print("demo_07 end")
    
    # 模拟生成数据
        # 模拟DCA1000EVM并使用AAnet预测手势
    def demo_09(self):
        
        # 初始化队列
        udp_queue = mp.Manager().Queue()
        frame_queue = mp.Manager().Queue()
        status = mp.Manager().Value('b', True)
        
        # 创建实时数据采集器
        collector = RealTimeCollector(ip_address='127.0.0.1', port=4098, output_file='data.bin',status=status,is_bind=False)
        
        # 创建DC1000EVM模拟器进程
        dca1000evm_simulator_process = mp.Process(target=collector.dca1000evm_simulator, args=(
            # "K:/手势识别数据集/2/2_3_Raw_0.bin",
            # "K:/手势识别数据集/2/2_5_Raw_0.bin",
            # "I:/aio/aio_radar/aio_gusture/dataset/2023_9_11/水杯快速向前向后",
            # "I:/aio/aio_radar/aio_gusture/dataset/2023_9_17/4",
            "K:/dataset/2024_3_7/2",
            # "K:/手势识别数据集/2/",
            10,
            '127.0.0.1', 
            4098,
            10,
            True,
            ))
        
        dca1000evm_simulator_process.start()       # 启动DC1000EVM模拟器

        # 等待进程结束
        dca1000evm_simulator_process.join()
        
        print("剩余未处理udp数据包数量：",udp_queue.qsize())
        print("剩余未处理帧数据数量：",frame_queue.qsize())
        print("demo_01 end")

if __name__ == "__main__":
    pass
    demo = Demo()
    # demo.demo_01()
    # demo.demo_02()
    # demo.demo_03()
    # demo.demo_04()
    # demo.demo_05()
    # demo.demo_06()
    # demo.demo_07()
    # demo.demo_08()
    demo.demo_09()