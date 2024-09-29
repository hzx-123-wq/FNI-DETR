from ultralytics import RTDETR

# Load a model
model = RTDETR("ultralytics/cfg/models/rt-detr/rtdetr-sp-mb-elans-l.yaml")  # build a new model from scratch
if __name__ == '__main__':

    # Use the model
    model.train(data="ultralytics/cfg/datasets/coco128.yaml", cfg="ultralytics/cfg/default.yaml", epochs=100)  # train the model