from ultralytics import RTDETR

model = RTDETR("runs/detect/only-aifimb-coco/weights/best1.pt")


if __name__ == '__main__':

    # Use the model
    model.val(data="ultralytics/cfg/datasets/VisDrone.yaml", cfg="ultralytics/cfg/default.yaml", epochs=1)  # train the model
