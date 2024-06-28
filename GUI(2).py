from PyQt5.QtWidgets import QApplication,QMainWindow,QFileDialog
import sys
from PyQt5.QtCore import Qt
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import QImage, QPixmap
import torch
import torchvision
import cv2
import numpy as np

from ultralytics import YOLO




class Ui_MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.retranslateUi(self)
    def setupUi(self, MainWindow): # 设置界面的组件，包括主窗口、按钮、标签等
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1128, 1009)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton.setGeometry(QtCore.QRect(20, 10, 93, 28))
        self.pushButton.setObjectName("pushButton")
        self.pushButton_2 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_2.setGeometry(QtCore.QRect(160, 10, 93, 28))
        self.pushButton_2.setObjectName("pushButton_2")
        self.pushButton_3 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_3.setGeometry(QtCore.QRect(290, 10, 93, 28))
        self.pushButton_3.setObjectName("pushButton_3")
        self.pushButton_4 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_4.setGeometry(QtCore.QRect(420, 10, 93, 28))
        self.pushButton_4.setObjectName("pushButton_4")
        self.label1 = QtWidgets.QTextBrowser(self.centralwidget)
        self.label1.setGeometry(QtCore.QRect(20, 60, 1071, 71))
        self.label1.setObjectName("label1")       
        self.label3 = QtWidgets.QLabel(self.centralwidget)
        self.label3.setGeometry(QtCore.QRect(320, 150, 481, 421))  # 将label3放到界面正中间        
        self.label3.setObjectName("label3")
        self.label3.setStyleSheet("border: 2px solid black;")  # 添加这行代码
        self.targetInfoBrowser = QtWidgets.QTextBrowser(self.centralwidget)
        self.targetInfoBrowser.setGeometry(QtCore.QRect(320, 575, 481, 421))  # 将targetInfoBrowser放到界面的下方
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
        self.pushButton.clicked.connect(self.uploadImage)
        self.pushButton_2.clicked.connect(self.showEnvironment)
        self.pushButton_3.clicked.connect(self.startProgram)
        self.pushButton_4.clicked.connect(self.uploadVideo)
        # self.image_path = ''

    def retranslateUi(self, MainWindow):# 设置界面各个组件的文本内容。
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.pushButton.setText(_translate("MainWindow", "上传图片"))
        self.pushButton_2.setText(_translate("MainWindow", "显示环境"))
        self.pushButton_3.setText(_translate("MainWindow", "启动程序"))
        self.pushButton_4.setText(_translate("MainWindow", "上传视频"))
        

    def uploadImage(self):
        file_dialog = QFileDialog()
        image_path, _ = file_dialog.getOpenFileName(self, '选择图片', '', 'Images (*.png *.xpm *.jpg *.bmp)')
        self.image_path = image_path
        if image_path:
            # 重置计数字典和文本框
            self.counts = {}
            # 清空targetInfoBrowser的内容
            self.targetInfoBrowser.clear()
            # 显示选择的图片
            pixmap = QPixmap(image_path)
            self.label3.setPixmap(pixmap)
            self.label3.setScaledContents(True)

    def uploadVideo(self):
        file_dialog = QFileDialog()
        video_path, _ = file_dialog.getOpenFileName(self, '选择视频', '', 'Videos (*.mp4 *.avi)')
        if video_path:
            # 重置计数字典和文本框
            self.counts = {}
            # 清空targetInfoBrowser的内容
            self.targetInfoBrowser.clear()
            self.cap = cv2.VideoCapture(video_path)
            fps = self.cap.get(cv2.CAP_PROP_FPS)
            self.timer = QtCore.QTimer(self)
            self.timer.timeout.connect(self.displayFrames)
            self.timer.start(int(1000 / fps))

    def displayFrames(self):
        batch_size = 4  # 设置批次大小
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
            model = YOLO('./runs/detect/train18/weights/best.pt', classes=[7,8,11,12,17], iou=0.7)
            results = model(frames, classes=[7,8,11,12,17], iou=0.7)  # 一次性处理整个批次

            for i, result in enumerate(results):
                # 更新每个类别的计数
                for box_data in result.boxes.data:
                    # 获取物体的类别标签
                    label = int(box_data[-1])
                    # 获取物体的类别名称
                    class_name = result.names[label]
                    # 如果这个类别还没有被计数过，那么在字典中添加一个新的条目
                    if class_name not in self.counts:
                        self.counts[class_name] = 0
                    # 增加计数
                    self.counts[class_name] += 1

                # 显示每个类别的总计数
                self.targetInfoBrowser.clear()
                for name, count in self.counts.items():
                    if count > 0:
                        self.targetInfoBrowser.append(f"<font size='5'>检测到的物体：{name}，数量：{count}</font><br>")

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
    
    def showEnvironment(self):
        pytorch_version = torch.__version__
        torchvision_version = torchvision.__version__
        self.label1.setText(f"PyTorch Version: {pytorch_version}\n"
                            f"Torchvision Version: {torchvision_version}")

    def startProgram(self):
        model = YOLO('./runs/detect/train18/weights/best.pt')
        results = model(self.image_path, classes=[7,8,11,12,17], iou=0.7)
        
        result = results[0]
        names = result.names
        
        # 创建一个字典来存储每个类别的计数
        counts = {name: 0 for name in names.values()}
        
        # 遍历每个检测到的物体
        for box_data in result.boxes.data:
            # 获取物体的类别标签
            label = int(box_data[-1])
            # 获取物体的类别名称
            class_name = names[label]
            # 增加计数
            counts[class_name] += 1
        
        annotated_frame = results[0].plot()
        annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        
        # 将图像数据转换为QImage格式
        height, width, channel = annotated_frame.shape
        bytes_per_line = 3 * width
        qimage = QImage(annotated_frame.data, width, height, bytes_per_line, QImage.Format_RGB888)
        # 将QImage转换为QPixmap
        pixmap = QPixmap.fromImage(qimage)

        # 用处理后的图片覆盖原图片      
        self.label3.setPixmap(pixmap)
        self.label3.setScaledContents(True)  
        
        # 显示每个类别的计数
        for name, count in counts.items():
            # 如果数量为0，则跳过
            if count == 0:
                 continue
            self.targetInfoBrowser.append(f"<font size='5'>检测到的物体：{name}，数量：{count}</font><br>")
        
        
               
if __name__ == '__main__':
    app = QApplication(sys.argv)
    MainWindow1 = QMainWindow()     #MainWindow1随便改
    ui = Ui_MainWindow()             #随便改
    ui.setupUi(MainWindow1)
    MainWindow1.show()
    sys.exit(app.exec_())
