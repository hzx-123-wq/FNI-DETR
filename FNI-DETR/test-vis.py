import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('runs/train-VisDrone/FNI-DETR/weights/best.pt')
    model.val(data='ultralytics/cfg/datasets/VisDrone.yaml',
              split='test',
              imgsz=640,
              batch=1,
              # iou=0.7,
              # rect=False,
              # save_json=True, # if you need to cal coco metrice
              project='runs/test-vis',
              name='FNI-DETR',
              )
