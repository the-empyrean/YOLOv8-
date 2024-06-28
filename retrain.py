from ultralytics import YOLO
# import os
# os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
# loss.py line 230

# model = YOLO("yolov8s.yaml").load("yolov8s.pt")
model = YOLO("runs/detect/train_s_data_5/weights/last.pt")

# model.train(data="coco.yaml", epochs=3)


if __name__ == '__main__':
    model.train(data = "dataset.yaml",val=True, epochs=100, batch=-1, patience=30, save=True, save_period=5, verbose=True, name='train_s_data_5')
    # save=True, save_period=5, verbose=True, name=
    # model.val()
    # model.save("my_yolov8n.pt")
    # save_dir