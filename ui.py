# 标准库
import os
import pickle
import subprocess
import tkinter as tk
import tkinter.filedialog as filedialog

# 第三方库
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import numpy as np
import scipy.io as sio
import torch
from tkinter import ttk
import matlab.engine


# 本地库
from RealTimeCollector.RealTimeCollector import RealTimeCollector
from ABCnet import RadarGestureNet
from ABCnet import one_hot_to_label


mpl.use('TkAgg')        # 启用tkinter渲染matplotlib，从而可以嵌入到tkinter中


class FileExplorerApp:
    
    def __init__(self, root):
        self.root = root
        self.root.title("采集数据工具")
        self.root.geometry("1200x700")

        self.rtc = RealTimeCollector(ip_address='127.0.0.99')
        self.cnt = 1
        self.start = 0
        self.model = None

        self.default_folder_path = r'D:\AA'
        self.default_model_path = r'D:\interface\aio_radar-main\lightning_logs'
        self.selected_folder = tk.StringVar()
        self.load_selected_folder()
        self.selected_model = tk.StringVar()
        self.load_selected_model()

        self.create_left_frame()
        self.create_right_frame()

        # 绑定关闭事件处理
        self.root.protocol("WM_DELETE_WINDOW", self.close_window)
        
        self.show_files(self.selected_folder.get())
        

    # 创建左侧框架
    def create_left_frame(self):
        """
        功能: 选择文件夹路径, 显示文件列表
        
        """
        
        left_frame = tk.Frame(self.root, width=400, height=600)
        left_frame.grid(row=0, column=0)

        # 文件夹路径选择器
        folder_label = tk.Label(left_frame, text="文件夹路径:")
        folder_label.grid(row=0, column=0, padx=10, pady=10)

        self.folder_entry = tk.Entry(left_frame, textvariable=self.selected_folder, width=40)
        self.folder_entry.grid(row=0, column=1, padx=10, pady=10)

        browse_button = tk.Button(left_frame, text="查找", command=self.browse_folder)
        browse_button.grid(row=0, column=2, padx=10, pady=10)

        # 文件列表
        file_list_label = tk.Label(left_frame, text="文件:")
        file_list_label.grid(row=1, column=0, padx=10, pady=10)

        self.file_listbox = tk.Listbox(left_frame, width=50, height=20)
        self.file_listbox.grid(row=2, column=0, columnspan=3, padx=10, pady=10)
        self.file_listbox.bind("<ButtonRelease-1>", self.show_selected_file)

        # 文件路径
        self.file_path_entry = tk.Entry(left_frame, width=50)
        self.file_path_entry.grid(row=4, column=1, padx=0, pady=0)

    # 创建右侧框架
    def create_right_frame(self):
        right_frame = tk.Frame(self.root, width=600, height=600)
        right_frame.grid(row=0, column=1)

        # 顶部框架
        upper_frame = tk.Frame(right_frame)
        upper_frame.grid(row=0, column=0, pady=10,columnspan=10)

        frame_label = tk.Label(upper_frame, text="设置", font=("Arial", 14))
        frame_label.grid(row=0, column=0, pady=5, columnspan=10)

        self.frame_labels = ["帧数:", "当前采集数据类别数:", "帧率:", "当前手势序号:"]
        self.entries = []
        for i, label_text in enumerate(self.frame_labels):
            self.label = tk.Label(upper_frame, text=label_text)
            self.label.grid(row=i+1, column=0, padx=5, pady=5, sticky="w")

            self.entry = tk.Entry(upper_frame, width=5)
            self.entry.grid(row=i+1, column=1,padx=5, pady=5)
            self.entries.append((self.label,self.entry))
        self.entries[3][1].insert(0,self.cnt)

        # 模型文件夹路径选择器
        model_label = tk.Label(upper_frame, text="模型路径:")
        model_label.grid(row=2, column=2, padx=10, pady=10)

        self.model_entry = tk.Entry(upper_frame, textvariable=self.selected_model, width=30)
        self.model_entry.grid(row=2, column=3, padx=10, pady=10)

        browse_button = tk.Button(upper_frame, text="选择", command=self.browse_model)
        browse_button.grid(row=2, column=4, padx=10, pady=10)

        # 模型预测
        self.model_label = tk.Label(upper_frame, text="模型预测结果:", font=("Arial", 12))
        self.model_label.grid(row=3, column=2, columnspan=2, pady=5)

        collect_button = tk.Button(upper_frame, text="模型预测",command=self.hangle_data)
        collect_button.grid(row=4, column=2,columnspan=2, padx=10, pady=5)

        # 实时模型检测复选框
        self.prediction_checkbox_var = tk.BooleanVar()
        self.prediction_checkbox = tk.Checkbutton(upper_frame, variable=self.prediction_checkbox_var,text="实时模型检测",command=self.model_prediction_checkbox)
        self.prediction_checkbox.grid(row=4, column=4, padx=10, pady=5)

        # 底部框架
        bottom_frame = tk.Frame(right_frame)
        bottom_frame.grid(row=1, column=0, columnspan=12,pady=10)

        # 图表显示区
        tab_control = ttk.Notebook(bottom_frame, width=580, height=400)
        tab_control.grid(row=0, column=0,rowspan=5,columnspan=10)

        self.tab_range = ttk.Frame(tab_control)
        tab_control.add(self.tab_range, text="Range")
        self.tab_speed = ttk.Frame(tab_control)
        tab_control.add(self.tab_speed, text="Speed")
        self.tab_angle = ttk.Frame(tab_control)
        tab_control.add(self.tab_angle, text="Angle")

        # 手势序号按钮
        prev_button = tk.Button(bottom_frame, text="上一个手势",command=self.previous_gesture)
        prev_button.grid(row=1, column=10, padx=10, pady=5)

        next_button = tk.Button(bottom_frame, text="下一个手势",comman=self.next_gesture)
        next_button.grid(row=2, column=10, padx=10, pady=5)


        # 启动及连接studio和matlab
        initialize_button = tk.Button(bottom_frame,text="启动",command=self.start_DCA)
        initialize_button.grid(row=3, column=10, padx=10, pady=5)

        initialize_button = tk.Button(bottom_frame,text="连接",command=self.initialization_radar)
        initialize_button.grid(row=4, column=10, padx=10, pady=5)



        # 数据预览复选框
        self.preview_checkbox_var = tk.BooleanVar()
        self.preview_checkbox = tk.Checkbutton(bottom_frame, variable=self.preview_checkbox_var,text="数据预览",command=self.handle_data_checkbox)
        self.preview_checkbox.grid(row=6, column=9, padx=10, pady=5)

        # 杂波滤除方式选择框
        self.clutter_selecter = tk.IntVar()
        self.radio_button1 = tk.Radiobutton(bottom_frame,variable=self.clutter_selecter,text="none",value=1,command=self.handle_data_checkbox)
        self.radio_button1.grid(row=6, column=0, padx=10, pady=5)
        self.radio_button2 = tk.Radiobutton(bottom_frame,variable=self.clutter_selecter,text="avg",value=2,command=self.handle_data_checkbox)
        self.radio_button2.grid(row=6, column=1, padx=10, pady=5)
        self.radio_button3 = tk.Radiobutton(bottom_frame,variable=self.clutter_selecter,text="mti",value=3,command=self.handle_data_checkbox)
        self.radio_button3.grid(row=6, column=2, padx=10, pady=5)
        self.clutter_selecter.set(1)

        # 采集数据按钮
        collect_button = tk.Button(bottom_frame, text="采集数据",command=self.collect_data)
        collect_button.grid(row=6, column=10, padx=10, pady=5)

    def browse_folder(self):
        """
        功能:
            打开文件夹对话框, 选择文件夹路径        
        用法:
            打开一个文件对话框来浏览和选择一个文件夹。 
        """
        folder_path = tk.filedialog.askdirectory(initialdir=self.default_folder_path)
        if folder_path:
            self.selected_folder.set(folder_path)
            self.save_selected_folder()
            self.show_files(folder_path)

    def show_files(self, folder_path):
        """
        功能:
            显示文件夹中的文件
        """
        self.file_listbox.delete(0, tk.END)
        self.file_paths = []  # 存储文件路径
        for file_name in os.listdir(folder_path):
            if file_name.endswith(".bin"):
                file_path = folder_path+"/"+file_name
                self.file_paths.append(file_path)  # 将文件路径添加到列表中
                self.file_listbox.insert(tk.END, file_name)

    def show_selected_file(self, event):
        selected_index = self.file_listbox.curselection()
        if selected_index:
            index = selected_index[0]
            file_path = self.file_paths[index]  # 获取选定文件的路径
            self.file_path_entry.delete(0, tk.END)
            self.file_path_entry.insert(tk.END, file_path)
            if self.preview_checkbox_var.get() or self.prediction_checkbox_var.get():
                self.hangle_data(file_path)

    def handle_data_checkbox(self):
        if self.preview_checkbox_var.get() or self.prediction_checkbox_var.get():
            filepath = self.file_path_entry.get()
            self.hangle_data(filepath)
        # if self.prediction_checkbox_var.get():
        #     self.model_prediction()
    
    def model_prediction_checkbox(self):
        if self.prediction_checkbox_var.get():
            # 如果模型为None, 则根据路径加载模型
            if self.model is None:
                model_path = self.model_entry.get()
                self.model = RadarGestureNet.load_from_checkpoint(model_path).to("cpu")
            
            self.hangle_data(self.file_path_entry.get())

    def save_selected_folder(self):
        with open("selected_folder.pkl", "wb") as file:
            pickle.dump(self.selected_folder.get(), file)

    def save_selected_model(self):
        with open("selected_model.pkl", "wb") as file:
            pickle.dump(self.selected_model.get(), file)

    def load_selected_folder(self):
        try:

            with open("selected_folder.pkl", "rb") as file:
                self.selected_folder.set(pickle.load(file))
        except FileNotFoundError:
            pass

    def load_selected_model(self):
        try:

            with open("selected_model.pkl", "rb") as file:
                self.selected_model.set(pickle.load(file))
        except FileNotFoundError:
            pass

    def previous_gesture(self):
        self.cnt = int(self.entries[3][1].get())
        if self.cnt > 1:
            self.cnt -= 1
            self.entries[3][1].delete(0,tk.END)
            self.entries[3][1].insert(0,self.cnt)

    def next_gesture(self):
        self.cnt = int(self.entries[3][1].get())
        self.cnt += 1
        self.entries[3][1].delete(0,tk.END)
        self.entries[3][1].insert(0,self.cnt)

    def start_DCA(self):
        dir_path = "mmwave_studio_02_01_01_00\\mmWaveStudio\\RunTime"
        # 打开CMD并执行命令
        subprocess.Popen("mmWaveStudio.exe /lua ..\\..\\..\\Scripts\\lua\\AIOStart.lua", cwd=dir_path, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)

    def initialization_radar(self):
        self.eng = matlab.engine.start_matlab()  # 启动matlab引擎

        # #!!!!
        # self.start == True
        # self.eng.cd(r'Scripts\matlab', nargout=0)
        # self.eng.main(nargout=0)

        if self.start == False:
            self.start = True
            # 连接到MATLAB引擎
            try:
                #将MATLAB的当前工作目录更改为指定的路径
                self.eng.cd(r'Scripts\matlab', nargout=0)
                #调用MATLAB中的main文件
                self.eng.main(nargout=0)
            except matlab.engine.MatlabExecutionError as e:
                print("MATLAB Error:", e)
            finally:
                # eng.quit()
                pass
        else:
            self.eng.cd(r'Scripts\matlab', nargout=0)
            self.eng.main(nargout=0)
        
    def browse_model(self):
        """
        功能:
            打开文件夹对话框, 选择模型路径        
        用法:
            打开一个文件对话框来浏览和选择一个模型。 
        """
        model_path = tk.filedialog.askopenfilename(initialdir=self.default_model_path)
        if model_path:
            pass
            self.selected_model.set(model_path)
            if model_path is None:
                # model_path = r'K:\aio_radar\lightning_logs\version_45\checkpoints\epoch=74-step=75.ckpt'
                model_path = r'D:\interface\aio_radar-main\lightning_logs\version_0\checkpoints\epoch=74-step=225.ckpt'
            self.model = RadarGestureNet.load_from_checkpoint(model_path).to("cpu")
            self.save_selected_model()

    def collect_data(self):
        self.cnt = int(self.entries[3][1].get())
        # file_path = f"K:\\\\aio_radar\\\\out\\\\{cnt}.bin"
        # file_path = f"K:\\\\dataset\\\\2024_3_11\\\\3\\\\{cnt}.bin"
        # file_path = f"D:\\\\interface\\\\dataset\\\\1\\\\{self.cnt}.bin"
        file_path = f"{self.folder_entry.get()}\\\\{self.cnt}.bin"
        file_path = file_path.replace("/", "\\\\")
        
        self.eng.start_record(file_path, nargout=2)
        self.cnt += 1
        self.entries[3][1].delete(0,tk.END)
        self.entries[3][1].insert(0,self.cnt)

        # 刷新文件夹内容
        self.file_listbox.delete(0, tk.END)
        self.file_paths = []  # 存储文件路径
        for file_name in os.listdir(self.folder_entry.get()):
            if file_name.endswith(".bin"):
                file_path = self.folder_entry.get()+"/"+file_name
                self.file_paths.append(file_path)  # 将文件路径添加到列表中
                self.file_listbox.insert(tk.END, file_name)

    def model_prediction(self):
        file_path = self.file_path_entry.get()
        model_path = self.model_entry.get()
        
        if self.clutter_selecter.get() == 1:
            clutter_removal = None
        elif self.clutter_selecter.get() == 2:
            clutter_removal = 'avg'
        else:
            clutter_removal = 'mtl'

        range_profile, speed_profile, angle_profile = self.rtc.process_file(file_path,clutter_removal)

        # 压缩维度
        range_profile = torch.tensor(np.squeeze(np.mean(np.abs(range_profile), axis=(0, 2))).T, dtype=torch.float32)
        speed_profile = torch.tensor(np.squeeze(np.mean(np.abs(speed_profile), axis=(0, 1))).T, dtype=torch.float32)
        angle_profile = torch.tensor(np.squeeze(np.mean(np.abs(angle_profile), axis=(1, 2))).T, dtype=torch.float32)



        # 预测
        # 在第0维增加一个维度(如果是二维的)
        if len(range_profile.shape) == 2:
            range_profile = range_profile.unsqueeze(0)
        if len(speed_profile.shape) == 2:
            speed_profile = speed_profile.unsqueeze(0)
        if len(angle_profile.shape) == 2:
            angle_profile = angle_profile.unsqueeze(0)
        
        out = self.model(range_profile, speed_profile, angle_profile)
        confidence = out[0][one_hot_to_label(out)].item()
        print("Confidence:", confidence)
        confidence_str = str(one_hot_to_label(out))
        if confidence > 0.4:
            print(out)
            print("Prediction:", one_hot_to_label(out), "!!!!!!!!!!!!!!!!!!")
            
            predictions = ["0", "1", "2", "3"]

            directions = {
                "0": "向左",
                "1": "向右",
                "2": "向上",
                "3": "向下",
            }

            for prediction in predictions:
                if prediction in confidence_str:
                    print(directions[prediction])
                    self.model_label.config(text="模型预测结果：" + directions[prediction])
        else:
            self.model_label.config(text="模型预测结果：无法识别")


    def hangle_data(self,file_path):

        if self.clutter_selecter.get() == 1:
            clutter_removal = None
        elif self.clutter_selecter.get() == 2:
            clutter_removal = 'avg'
        else:
            clutter_removal = 'mtl'

        range_profile, speed_profile, angle_profile = self.rtc.process_file(file_path,clutter_removal)

        # 压缩维度
        range_profile = np.squeeze(np.mean(np.abs(range_profile), axis=(0, 2))).T
        speed_profile = np.squeeze(np.mean(np.abs(speed_profile), axis=(0, 1))).T
        angle_profile = np.squeeze(np.mean(np.abs(angle_profile), axis=(1, 2))).T

        # 如果数据预览复选框被选中, 则显示数据
        if self.preview_checkbox_var.get():
        
            # 创建一个Figure对象
            plt.close()
            self.fig1, self.ax1 = plt.subplots(figsize=(6, 4))
            self.ax1.imshow(abs(range_profile), cmap='jet', aspect='auto')
            self.ax1.set_xlabel('距离 / m')
            self.ax1.set_ylabel('时间 / s')

            self.fig2, self.ax2 = plt.subplots(figsize=(6, 4))
            self.ax2.imshow(abs(speed_profile), cmap='jet', aspect='auto')
            self.ax2.set_xlabel('速度 / m')
            self.ax2.set_ylabel('时间 / s')

            self.fig3, self.ax3 = plt.subplots(figsize=(6, 4))
            self.ax3.imshow(abs(angle_profile), cmap='jet', aspect='auto')
            self.ax3.set_xlabel('角度 / m')
            self.ax3.set_ylabel('时间 / s')

            plt.close(self.fig1)
            plt.close(self.fig2)
            plt.close(self.fig3)


            # 创建一个FigureCanvasTkAgg对象
            self.canvas1 = FigureCanvasTkAgg(self.fig1, master=self.tab_range)
            self.canvas1.draw()
            self.canvas1.get_tk_widget().grid(row=0, column=0)
            self.canvas2 = FigureCanvasTkAgg(self.fig2, master=self.tab_speed)
            self.canvas2.draw()
            self.canvas2.get_tk_widget().grid(row=0, column=0)
            self.canvas3 = FigureCanvasTkAgg(self.fig3, master=self.tab_angle)
            self.canvas3.draw()
            self.canvas3.get_tk_widget().grid(row=0, column=0)
        
        if self.prediction_checkbox_var.get():
            
            # 转化为tensor
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
            
            out = self.model(range_profile, speed_profile, angle_profile)
            confidence = out[0][one_hot_to_label(out)].item()
            print("Confidence:", confidence)
            print(out)
            confidence_str = str(one_hot_to_label(out))
            if confidence > 0.4:
                
                print("Prediction:", one_hot_to_label(out), "!!!!!!!!!!!!!!!!!!")
                predictions = ["0", "1", "2", "3"]

                directions = {
                    "0": "向左",
                    "1": "向右",
                    "2": "向上",
                    "3": "向下",
                }

                for prediction in predictions:
                    if prediction in confidence_str:
                        print(directions[prediction])
                        self.model_label.config(text="模型预测结果：" + directions[prediction])
            else:
                self.model_label.config(text="模型预测结果：无法识别")


    def close_window(self):
        # 关闭窗口时结束Tkinter的事件循环
        self.root.quit()



if __name__ == "__main__":
    root = tk.Tk()
    app = FileExplorerApp(root)
    root.mainloop()
