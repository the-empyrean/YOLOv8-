from PyQt5.QtWidgets import QApplication,QMainWindow,QFileDialog
import sys
from PyQt5.QtCore import Qt
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QApplication
import torch
import torchvision
import cv2
import numpy as np
import glob
from collections import defaultdict
from ultralytics import YOLO
import pandas as pd

label2conf = {'YiJia':0.3, 'ShuPian':0.3, 'PingZhuangYinYongShui':0.5, 'GuanZhuangYinLiao':0.55, 'PingGuo':0.65}
label2id = {'YiJia':'000', 'ShuPian':'001', 'PingZhuangYinYongShui':'002', 'PingGuo':'003', 'GuanZhuangYinLiao':'004'}

class Ui_MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.retranslateUi(self)
        self.frame_counts = defaultdict(list)  # 存储每个类别在每一帧中的计数
        self.frames = []  # 存储批次的帧
    def setupUi(self, MainWindow): # 设置界面的组件，包括主窗口、按钮、标签等
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1128, 700)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton.setGeometry(QtCore.QRect(150, 75, 200, 100))  # 改变按钮的大小和位置
        self.pushButton.setObjectName("pushButton")
        self.pushButton_4 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_4.setGeometry(QtCore.QRect(700, 75, 200, 100))  # 改变按钮的大小和位置
        self.pushButton_4.setObjectName("pushButton_4")
        self.label3 = QtWidgets.QLabel(self.centralwidget)
        self.label3.setGeometry(QtCore.QRect(50, 200, 481, 421))  # 将label3放到界面的左边      
        self.label3.setObjectName("label3")
        self.label3.setStyleSheet("border: 2px solid black;")  # 添加这行代码
        self.targetInfoBrowser = QtWidgets.QTextBrowser(self.centralwidget)
        self.targetInfoBrowser.setGeometry(QtCore.QRect(550, 200, 481, 421))  # 将targetInfoBrowser放到界面的右边
        self.targetInfoBrowser.setObjectName("targetInfoBrowser")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1128, 26))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)


        # 点击响应函数
        self.pushButton.clicked.connect(self.imageRecognition)       
        self.pushButton_4.clicked.connect(self.uploadVideo)
        # self.image_path = ''

    def retranslateUi(self, MainWindow):# 设置界面各个组件的文本内容。
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        font = QtGui.QFont()
        font.setPointSize(20)  # 设置字体大小为20
        self.pushButton.setFont(font)  # 设置字体
        self.pushButton.setText(_translate("MainWindow", "图片识别"))
        self.pushButton_4.setFont(font)  # 设置字体
        self.pushButton_4.setText(_translate("MainWindow", "视频识别"))
        
    def imageRecognition(self):
        # 获取所有的图片路径，并按照其所在的组进行排序
        image_paths = sorted(glob.glob('data/r1/*/*.jpg'), key=lambda x: x.split('/')[-2])

        group_results = defaultdict(dict)
        temp_results = defaultdict(list)

        for i, image_path in enumerate(image_paths):
            group_name = f'group_{i // 2 + 1}'

            # 重置计数字典和文本框
            self.counts = {}
            # 清空targetInfoBrowser的内容
            self.targetInfoBrowser.clear()
            
            model = YOLO('runs/detect/train_s_data_8_new/weights/epoch50.pt')
            results = model(image_path,
                            classes=[7, 8, 11, 12, 17],
                            iou=0.5,
                            )
            
            result = results[0]
            names = result.names
            # names = {0:'YiJia', 1:'ShuPian', 2:'PingZhuangYinYongShui', 3:'PingGuo', 4:'GuanZhuangYinLiao'}
            # print(names)

            # 创建一个字典来存储每个类别的计数
            counts = {name: 0 for name in names.values()}
            
            # 遍历每个检测到的物体
            # for box_data in result.boxes.data:
            #     # 获取物体的类别标签
            #     label = int(box_data[-1])
            #     # 获取物体的类别名称
            #     class_name = names[label]
            #     # 增加计数
            #     counts[class_name] += 1

            for conf, box_data in zip(result.boxes.conf, result.boxes.data):
                # 获取物体的类别标签
                label = int(box_data[-1])
                # 获取物体的类别名称
                class_name = names[label]
                if (conf < label2conf[class_name]):
                    continue
                # 增加计数
                counts[class_name] += 1
            
            # 显示每个类别的计数
            for name, count in counts.items():
                # 如果数量为0，则跳过
                if count == 0:
                     continue
                self.targetInfoBrowser.append(f"<font size='5'>{name}，数量：{count}</font><br>")
            
            # 将每张图片的结果存储在临时字典中
            for name, count in counts.items():
                temp_results[name].append(count)

            # 如果处理完一组图片（每两张图片为一组），将临时字典中每个类别的最大值存储在最终结果字典中
            if (i + 1) % 2 == 0:
                for name, counts in temp_results.items():
                    group_results[group_name][name] = max(counts)
                temp_results.clear()

            # 获取处理后的图像
            annotated_frame = results[0].plot()
            annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            # 将图像数据转换为QImage格式
            height, width, channel = annotated_frame.shape
            bytes_per_line = 3 * width
            qimage = QtGui.QImage(annotated_frame.data, width, height, bytes_per_line, QtGui.QImage.Format_RGB888)
            # 将QImage转换为QPixmap
            pixmap = QtGui.QPixmap.fromImage(qimage)
            # 显示处理后的图像
            self.label3.setPixmap(pixmap)
            self.label3.setScaledContents(True)


            # 处理所有的事件，确保界面更新
            QApplication.processEvents()

        # 打印每组的结果
        for group_name, counts in group_results.items():
            print(f'Group {group_name}: {counts}')
            print()
            
        # 计算所有组的总计数
        total_counts = defaultdict(int)
        for key in label2id.keys():
            total_counts[label2id[key]] = 0

        for counts in group_results.values():
            for name, count in counts.items():
                if name in label2id.keys():
                    total_counts[label2id[name]] += count

        
        # # 删除计数为0的项
        # total_counts = {name: count for name, count in total_counts.items() if count > 0}
        # 打印总计数
        print(f'Total: {total_counts}')

        # 创建一个DataFrame，其中的数据是total_counts字典
        df = pd.DataFrame.from_records([total_counts])
        # df.sort_index()
        # 将DataFrame保存为Excel文件
        # df.to_excel('C:/Users/Liu Chuan/Desktop/第1轮识别结果.xlsx', index=False)
        df.to_excel('C:/Users/Liu Chuan/Desktop/1e9+7队第1轮识别结果.xlsx', index=False)

    def uploadVideo(self):
        # 使用相对路径指定视频文件
        video_path = 'data/r2.mp4'
        # 重置计数字典和文本框
        self.counts = {}
        self.frame_counts = defaultdict(list)  # 初始化frame_counts
        # 清空targetInfoBrowser的内容
        self.targetInfoBrowser.clear()
        self.cap = cv2.VideoCapture(video_path)
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))  # 获取视频的总帧数
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.displayFrames)
        self.timer.start(int(1000 / fps))

    def displayFrames(self):
        batch_size = 8  # 设置批次大小
        frames = []  # 存储批次的帧

        # 读取批次的帧
        for _ in range(batch_size):
            ret, frame = self.cap.read()
            if ret:
                frames.append(frame)
            else:
                break

        if frames:
            # 使用 YOLO 模型检测物体
            model = YOLO('runs/detect/train_s_data_8_new/weights/epoch50.pt')
            results = model(frames,
                            classes=[7, 8, 11, 12, 17],
                            iou=0.5,
                            conf=0.3,
                            )  # 一次性处理整个批次

            for i, result in enumerate(results):
                # 创建一个新的字典来存储当前帧的计数
                current_counts = defaultdict(int)

                # 更新每个类别的计数
                for box_data in result.boxes.data:
                    # 获取物体的类别标签
                    label = int(box_data[-1])
                    # 获取物体的类别名称
                    class_name = result.names[label]
                    # 增加计数
                    current_counts[class_name] += 1

                # 将每一帧的计数添加到frame_counts字典中
                for name, count in current_counts.items():
                    self.frame_counts[name].append(count) 

                # 清空targetInfoBrowser的内容
                self.targetInfoBrowser.clear()
                # 显示每个类别的计数
                for name, count in current_counts.items():
                    if count > 0:
                        self.targetInfoBrowser.append(f"<font size='5'>{name}，数量：{count}</font><br>")

                annotated_frame = result.plot()
                annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                
                # 显示经过 YOLO 处理后的视频帧
                height, width, channel = annotated_frame.shape
                bytes_per_line = 3 * width
                qimage = QImage(annotated_frame.data, width, height, bytes_per_line, QImage.Format_RGB888)
                pixmap = QPixmap.fromImage(qimage)
                
                self.label3.setPixmap(pixmap)
                self.label3.setScaledContents(True)

        else:
            self.timer.stop()
            self.cap.release()
            self.calculateFinalCounts()

    def calculateFinalCounts(self):
        # 在处理所有帧之后，找出每个类别中出现次数最多的数量
        final_counts = {}
        total_frames = self.total_frames  # 总帧数
        for name, counts in self.frame_counts.items():
            count_frequency = defaultdict(int)
            # 计算出现次数为0的次数
            count_frequency[0] = total_frames - len(counts)
            for count in counts:
                count_frequency[count] += 1

            # 打印每个类别出现数量的出现次数
            print(f'类别 {name} 的出现数量的出现次数：{count_frequency}')

            # 找出出现次数最多的数量
            max_frequency = max(count_frequency.values())
            final_count = [count for count, frequency in count_frequency.items() if frequency == max_frequency][0]

            # 如果最多出现的次数是0，那么忽略这个类别
            if final_count > 0:
                final_counts[name] = final_count

        for label in label2id.keys():
            if label not in final_counts.keys():
                final_counts[label] = 0

        ff_counts = defaultdict(int)
        for label, num in final_counts.items():
            ff_counts[label2id[label]] = num

        print(f'最终的识别结果是：{ff_counts}')

        # 创建一个DataFrame，其中的数据是final_counts字典
        df = pd.DataFrame.from_records([ff_counts])
        # _ = [f'00{i}' for i in range(5)]
        # df = df[_]
        df = df.sort_index(axis=1)

        # 将DataFrame保存为Excel文件
        # df.to_excel('C:/Users/Liu Chuan/Desktop/第2轮识别结果.xlsx', index=False)
        df.to_excel('C:/Users/Liu Chuan/Desktop/1e9+7队第2轮识别结果.xlsx', index=False)
               
if __name__ == '__main__':
    app = QApplication(sys.argv)
    MainWindow1 = QMainWindow()     
    ui = Ui_MainWindow()             
    ui.setupUi(MainWindow1)
    MainWindow1.show()
    sys.exit(app.exec_())
