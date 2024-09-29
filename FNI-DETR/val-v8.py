from ultralytics import RTDETR

model = RTDETR("runs/detect/test/best.pt")


if __name__ == '__main__':

    # Use the model
    model.val(data="ultralytics/cfg/datasets/coco.yaml", cfg="ultralytics/cfg/default.yaml", epochs=1)  # train the model