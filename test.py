from ultralytics import YOLO

# Load a model
# model = YOLO('yolov8s.pt')  # load an official model
model = YOLO('runs/detect/train_s_data_6/weights/epoch5.pt')  # load a custom model


if __name__ == '__main__':
    # Validate the model
    metrics = model.val(split='val')  # no arguments needed, dataset and settings remembered
    metrics.box.map  # map50-95
    metrics.box.map50  # map50
    metrics.box.map75  # map75
    metrics.box.maps  # a list contains map50-95 of each category