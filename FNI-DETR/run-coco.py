from ultralytics import RTDETR

# Load a model
model = RTDETR("ultralytics/cfg/models/rt-detr/detr-all-l.yaml")  # build a new model from scratch
if __name__ == '__main__':

    # Use the model
    model.train(data="ultralytics/cfg/datasets/coco.yaml", 
                imgsz=640, 
                epochs=100, 
                batch=4, 
                workers=4, 
                project='runs/train-coco'
                name='FNI-DETR'
               )  # train the model
