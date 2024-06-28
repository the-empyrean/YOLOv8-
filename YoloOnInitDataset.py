from ultralytics import YOLO
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
# loss.py line 230

model = YOLO("yolov8n.yaml").load("yolov8n.pt")


# model.train(data="coco.yaml", epochs=3)


if __name__ == '__main__':
    model.train(data = "dataset.yaml",epochs=300,patience=30,save=True,save_period=1,verbose=True,name='train20_2')
    # save=True, save_period=5, verbose=True, name=
    model.val()
    model.save("my_yolov8n.pt")
    # save_dir