from PyQt5.QtWidgets import QApplication,QMainWindow,QFileDialog
import sys

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
        self.label2 = QtWidgets.QLabel(self.centralwidget)
        self.label2.setGeometry(QtCore.QRect(40, 190, 481, 421))
        self.label2.setObjectName("label2")
        self.label3 = QtWidgets.QLabel(self.centralwidget)
        self.label3.setGeometry(QtCore.QRect(600, 200, 461, 381))
        self.label3.setObjectName("label3")
        self.targetInfoBrowser = QtWidgets.QTextBrowser(self.centralwidget)
        self.targetInfoBrowser.setGeometry(QtCore.QRect(20, 600, 1080, 100))  # 设置位置和大小
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
        self.label2.setText(_translate("MainWindow", "TextLabel"))
        self.label3.setText(_translate("MainWindow", "TextLabel"))

    def uploadImage(self):
        file_dialog = QFileDialog()
        image_path, _ = file_dialog.getOpenFileName(self, '选择图片', '', 'Images (*.png *.xpm *.jpg *.bmp)')
        self.image_path = image_path
        if image_path:
            # 在这里添加加载图片的逻辑，例如显示图片到label2
            pixmap = QtGui.QPixmap(image_path)
            self.label2.setPixmap(pixmap)
            self.label2.setScaledContents(True)

    def uploadVideo(self):
        file_dialog = QFileDialog()
        video_path, _ = file_dialog.getOpenFileName(self, '选择视频', '', 'Videos (*.mp4 *.avi)')
        if video_path:
            self.cap = cv2.VideoCapture(video_path)
            fps = self.cap.get(cv2.CAP_PROP_FPS)
            self.timer = QtCore.QTimer(self)
            self.timer.timeout.connect(self.displayFrames)
            self.timer.start(int(1000 / fps))


    def displayFrames(self):
        ret, frame = self.cap.read()
        if ret:
            # 显示原视频
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, c = rgb_frame.shape
            qimg = QImage(rgb_frame.data, w, h, w * c, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qimg)
            self.label2.setPixmap(pixmap)
            self.label2.setScaledContents(True)

            # 使用 YOLO 模型检测物体
            model = YOLO('yolov8n.pt')
            results = model(frame)
            annotated_frame = results[0].plot()

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
        model = YOLO('yolov8n.pt')
        results = model(self.image_path)

        annotated_frame = results[0].plot()

        # 将图像数据转换为QImage格式
        height, width, channel = annotated_frame.shape
        bytes_per_line = 3 * width
        qimage = QImage(annotated_frame.data, width, height, bytes_per_line, QImage.Format_RGB888)
        # 将QImage转换为QPixmap
        pixmap = QPixmap.fromImage(qimage)

        # 设置显示图片的标签
        self.label3.setPixmap(pixmap)
        self.label3.setScaledContents(True)  
        
        
           
        
            
if __name__ == '__main__':
    app = QApplication(sys.argv)
    MainWindow1 = QMainWindow()     #MainWindow1随便改
    ui = Ui_MainWindow()             #随便改
    ui.setupUi(MainWindow1)
    MainWindow1.show()
    sys.exit(app.exec_())
