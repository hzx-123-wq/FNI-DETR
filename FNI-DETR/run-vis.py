from ultralytics import RTDETR

# Load a model
model = RTDETR("ultralytics/cfg/models/rt-detr/detr-all-l.yaml")  # build a new model from scratch
if __name__ == '__main__':

    # Use the model
    model.train(data="ultralytics/cfg/datasets/VisDrone.yaml", 
                imgsz=640, 
                epochs=100, 
                batch=4, 
                workers=4, 
                project='runs/train-VisDrone'
                name='FNI-DETR'
               )  # train the model
