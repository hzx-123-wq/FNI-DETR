import warnings
warnings.filterwarnings('ignore')
from ultralytics import RTDETR

if __name__ == '__main__':
    model = RTDETR('runs/train-coco/FNI-DETR/weights/best.pt')
    model.val(data='ultralytics/cfg/datasets/coco.yaml',
              split='val',
              imgsz=640,
              batch=1,
              line_width=1, 
            #   save_json=True, # if you need to cal coco metrice
              project='runs/val-coco',
              name='FNI-DETR',
              )
