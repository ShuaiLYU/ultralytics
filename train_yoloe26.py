

import ultralytics,os
workspace = os.path.dirname(os.path.dirname(os.path.abspath(ultralytics.__file__)))
os.chdir(workspace)
print("set workspace:", workspace)



from ultralytics import YOLOE,YOLO
from ultralytics.models.yolo.yoloe import YOLOETrainerFromScratch,YOLOEVPTrainer,YOLOEPEFreeTrainer
from ultralytics.models.yolo.yoloe import YOLOESegTrainerFromScratch #,YOLOESegTrainerSegHead


# model =YOLO("yoloe-26.yaml")


# model.train(
#     data="yoloe26_coco128.yaml",
#     epochs=2,
#     batch=4,
#     imgsz=640,
#     workers=4,
#     trainer=YOLOETrainerFromScratch)



model=YOLO("/Users/louis/workspace/ultra_louis_work/ultralytics/runs/detect/train6/weights/best.pt")


model.train(
    data="yoloe26_coco128.yaml",
    refer_data="coco128.yaml",
    epochs=2,
    batch=4,
    imgsz=640,
    workers=4,
    trainer=YOLOEVPTrainer)

