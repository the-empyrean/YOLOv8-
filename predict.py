from ultralytics import YOLO
import cv2
from PIL import Image


model = YOLO("./runs/detect/train_s_data_8_new/weights/epoch50.pt")  # load a pretrained model (recommended for training)
model.predict(source='Z:/aaa/my_data/images/data/r1/group3/',
              save=True,
              visualize=False,
              classes=[7,8,11,12,17],
              name='Z:/aaa/Yolo/predict/s_d8_new/data_g3/',
              # iou=0.5,
              # conf=0.5,
              )


