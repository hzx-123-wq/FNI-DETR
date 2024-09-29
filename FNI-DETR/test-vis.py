from ultralytics import YOLO
model = YOLO('v8.pt')
if __name__ == '__main__':
    model.predict("ultralytics/assets/bus6.jpg", imgsz=640, save=True, device=0)
